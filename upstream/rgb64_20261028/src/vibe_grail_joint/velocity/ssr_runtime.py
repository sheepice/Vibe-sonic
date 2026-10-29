"""Training-only SSR contact foresight and reward engine for the existing G1 task."""

import copy
import os
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import quat_apply

from .ssr_foothold import (
    ContactPredictor, ContactReplay, DelayedContacts, foothold_score, gaussian_nll,
    gaussian_quadrature, patch_offsets, rotate_xy, support_deficiency, yaw_angle,
)

FEATURE_DIM = 228


class SSRGuidance:
    def __init__(self, env, options):
        self.env, self.options = env, copy.deepcopy(options)
        self.c = env.command_manager.get_term("motion")
        self.ids = self.c.ppo_online.nonzero().flatten()
        self.feet = [list(self.c.cfg.body_names).index(f"{side}_ankle_roll_link")
                     for side in ("left", "right")]
        device, n = env.device, len(self.ids)
        self.patch = patch_offsets(options["sole_length_m"], options["sole_width_m"],
                                   options["patch_spacing_m"], device=device)
        self.foot_grid = torch.cartesian_prod(torch.linspace(-.15, .15, 5, device=device),
                                             torch.linspace(-.10, .10, 5, device=device))
        self.base_grid = torch.cartesian_prod(torch.linspace(-.6, .6, 7, device=device),
                                             torch.linspace(-.4, .4, 5, device=device))
        devices = [torch.device(device).index] if torch.device(device).type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(options["seed"])
            self.model = ContactPredictor(FEATURE_DIM, options["sigma_min_m"],
                                          options["sigma_max_m"]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=options["learning_rate"])
        rank = int(os.environ.get("RANK", "0"))
        self.replay = ContactReplay(options["replay_capacity"], FEATURE_DIM, device,
                                    options["seed"]+rank)
        self.pending = DelayedContacts(n, FEATURE_DIM, options["pending_slots"],
                                       options["max_contact_horizon_steps"], device)
        self.streak = torch.zeros(n, 2, device=device, dtype=torch.long)
        self.first_contact_xy = torch.zeros(n, 2, 2, device=device)
        self.update_count = 0
        self.prediction_rmse_ema = None
        self.last_reward_step = -1
        self.cached_score = torch.zeros(env.num_envs, device=device)
        self.metrics, self.training_metrics = {}, {}

    def synchronize_initial_model(self):
        if torch.distributed.is_initialized():
            for parameter in self.model.parameters():
                torch.distributed.broadcast(parameter.data, src=0)

    def contact_state(self):
        results = []
        for side in ("left", "right"):
            data = self.env.scene[f"foothold_{side}"].data
            results.append(((data.found > 0) & (data.normal[..., 2].abs() > .7)
                            & (data.force[..., 2].abs() > self.options["contact_force_n"])).any(-1))
        return torch.stack(results, -1)[self.ids]

    def sole_centers(self):
        pos = self.c.robot_body_pos_w[self.ids][:, self.feet]
        quat = self.c.robot_body_quat_w[self.ids][:, self.feet]
        offset = pos.new_tensor([self.options["sole_center_x_m"], 0.,
                                self.options["sole_center_z_m"]]).expand_as(pos)
        return pos+quat_apply(quat.reshape(-1, 4), offset.reshape(-1, 3)).reshape_as(pos), quat

    def features(self, actions, contact, centers):
        robot, ids = self.c.robot, self.ids
        root = robot.data.root_link_pos_w[ids]
        yaw = yaw_angle(robot.data.root_link_quat_w[ids])

        def local_velocity(value):
            return torch.cat([rotate_xy(value[..., :2], -yaw), value[..., 2:]], -1)

        foot_pos = centers-root[:, None]
        foot_pos = torch.cat([rotate_xy(foot_pos[..., :2], -yaw), foot_pos[..., 2:]], -1)
        foot_vel = local_velocity(self.c.robot_body_lin_vel_w[ids][:, self.feet])
        foot_xy = centers[..., None, :2]+rotate_xy(
            self.foot_grid.expand(len(ids), 2, -1, -1), yaw)
        base_xy = root[:, None, :2]+rotate_xy(self.base_grid.expand(len(ids), -1, -1), yaw)
        foot_h = self.c.ground_height(foot_xy, ids)-root[:, None, None, 2]
        base_h = self.c.ground_height(base_xy, ids)-root[:, None, 2]
        clearance = root[:, 2]-self.c.ground_height(root[:, :2], ids)
        command = torch.stack([self.c.speed[ids]*yaw.cos(), -self.c.speed[ids]*yaw.sin(),
                               torch.zeros_like(yaw)], -1)
        values = torch.cat([
            local_velocity(robot.data.root_link_lin_vel_w[ids]),
            local_velocity(robot.data.root_link_ang_vel_w[ids]),
            robot.data.projected_gravity_b[ids], command,
            robot.data.joint_pos[ids], robot.data.joint_vel[ids]/10.,
            self.env.action_manager.action[ids].clamp(-10., 10.)/5.,
            actions[ids].clamp(-10., 10.)/5.,
            foot_pos.flatten(1), foot_vel.flatten(1)/5., contact.float(),
            foot_h.flatten(1).clamp(-3., 3.), base_h.clamp(-3., 3.), clearance[:, None],
        ], -1)
        if values.shape != (len(ids), FEATURE_DIM) or not torch.isfinite(values).all():
            raise RuntimeError("Invalid SSR privileged feature layout or values")
        return values, root[:, :2], yaw

    def patch_heights(self, centers_xy, yaw, ids):
        # centers: [batch, feet, (Gaussian samples), XY].
        xy = centers_xy.unsqueeze(-2)+rotate_xy(
            self.patch.expand(*centers_xy.shape[:-1], -1, -1), yaw)
        return self.c.ground_height(xy, ids)

    @torch.no_grad()
    def before_step(self, actions):
        contact = self.contact_state()
        centers, quat = self.sole_centers()
        features, origins, base_yaw = self.features(actions, contact, centers)
        step = int(self.env.common_step_counter)
        if step % self.options["sample_interval_steps"] == 0:
            self.pending.add(features, origins, base_yaw, ~contact, step)
        mu, sigma = self.model(features)
        if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all():
            raise RuntimeError("Non-finite SSR imagined contact distribution")
        gaussian_xy, weights = gaussian_quadrature(mu, sigma)
        gaussian_xy = rotate_xy(gaussian_xy, base_yaw)+origins[:, None, None]
        foot_yaw = yaw_angle(quat)
        stance_deficit, predicted_deficit = [], []
        chunk = self.options["geometry_batch_envs"]
        # Bounded geometry temporaries: avoid [4080,2,9,40,16] intermediate arrays.
        for begin in range(0, len(self.ids), chunk):
            end = min(begin+chunk, len(self.ids))
            ids = self.ids[begin:end]
            stance_h = self.patch_heights(centers[begin:end, :, :2], foot_yaw[begin:end], ids)
            imagined_h = self.patch_heights(
                gaussian_xy[begin:end], foot_yaw[begin:end, :, None], ids)
            stance_deficit.append(support_deficiency(
                stance_h, centers[begin:end, :, 2], self.options["height_tolerance_m"]))
            predicted_deficit.append((support_deficiency(
                imagined_h, tolerance=self.options["height_tolerance_m"])*weights).sum(-1))
        stance_deficit, predicted_deficit = torch.cat(stance_deficit), torch.cat(predicted_deficit)
        ready = (self.update_count >= self.options["predictor_warmup_updates"]
                 and self.replay.seen >= self.options["minimum_labels"]
                 and self.prediction_rmse_ema is not None
                 and self.prediction_rmse_ema <= self.options["maximum_prediction_rmse_m"])
        level = self.c.tile_data["level"][self.c.tile[self.ids]]
        enabled = (level >= self.options["minimum_terrain_level"])[:, None] & ready
        score = foothold_score(contact, stance_deficit, predicted_deficit, enabled,
                                self.options["reward_variance"])
        warmup = min(1., (self.update_count+1)/self.options["reward_warmup_updates"])
        self.cached_score.zero_()
        self.cached_score[self.ids] = score*warmup
        self.metrics = {
            "SSR/stance_deficiency": stance_deficit[contact].mean() if contact.any() else score.new_zeros(()),
            "SSR/imagined_deficiency": predicted_deficit.mean(),
            "SSR/imagined_enabled_fraction": enabled.float().mean(),
            "SSR/reward": score.mean(), "SSR/predicted_sigma_m": sigma.mean(),
            "SSR/label_count": float(self.replay.seen), "SSR/warmup": warmup,
        }

    @torch.no_grad()
    def after_physics_reward(self):
        step = int(self.env.common_step_counter)
        if step == self.last_reward_step:
            return self.cached_score
        contact = self.contact_state()
        centers, _ = self.sole_centers()
        started = contact & (self.streak == 0)
        self.first_contact_xy[started] = centers[..., :2][started]
        self.streak = torch.where(contact, self.streak+1, 0)
        valid_contact = self.streak == self.options["valid_contact_steps"]
        ended = self.env.reset_buf[self.ids].bool()
        features, feet, targets = self.pending.resolve(
            valid_contact, self.first_contact_xy, ended, step)
        if len(features):
            # Online error on newly resolved, previously unseen labels; never a fitted minibatch score.
            mu, _ = self.model(features)
            predicted = mu[torch.arange(len(feet), device=feet.device), feet]
            rmse = float((predicted-targets).square().sum(-1).mean().sqrt())
            self.prediction_rmse_ema = (rmse if self.prediction_rmse_ema is None
                                       else .98*self.prediction_rmse_ema+.02*rmse)
            self.replay.add(features, feet, targets)
        self.metrics["SSR/online_prediction_rmse_m"] = (
            self.prediction_rmse_ema if self.prediction_rmse_ema is not None else 0.)
        self.metrics["SSR/label_count"] = float(self.replay.seen)
        self.env.extras.setdefault("log", {}).update(self.metrics)
        self.last_reward_step = step
        return self.cached_score

    def reset(self, env_ids):
        selected = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        if env_ids is not None:
            selected.zero_()
            selected[env_ids] = True
        mask = selected[self.ids]
        self.pending.reset(mask)
        self.streak[mask] = 0

    def update(self):
        """Separate optimizer; all ranks execute identical collectives, even with no labels."""
        losses = []
        with torch.inference_mode(False), torch.enable_grad():
            for _ in range(self.options["gradient_steps_per_update"]):
                self.optimizer.zero_grad(set_to_none=True)
                count = min(self.options["predictor_batch_size"], self.replay.size)
                if count:
                    features, feet, target = self.replay.sample(count)
                    mu, sigma = self.model(features)
                    row = torch.arange(count, device=feet.device)
                    loss = gaussian_nll(mu[row, feet], sigma[row, feet], target)
                else:
                    loss = sum(p.sum()*0. for p in self.model.parameters())
                loss.backward()
                parameters = list(self.model.parameters())
                gradients = torch.cat([p.grad.flatten() for p in parameters])
                weight = gradients.new_tensor(float(count > 0))
                distributed = torch.distributed.is_initialized()
                if distributed:
                    torch.distributed.all_reduce(gradients)
                    torch.distributed.all_reduce(weight)
                gradients /= weight.clamp(min=1.)
                if not torch.isfinite(gradients).all():
                    raise RuntimeError("Non-finite SSR predictor gradient")
                offset = 0
                for p in parameters:
                    p.grad.copy_(gradients[offset:offset+p.numel()].reshape_as(p))
                    offset += p.numel()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                if float(weight) > 0:
                    self.optimizer.step()
                losses.append(float(loss.detach()))
        self.update_count += 1
        self.training_metrics = dict(ssr_prediction_nll=sum(losses)/max(1, len(losses)))
        return self.training_metrics

    def save(self, directory, iteration, rank, world):
        state = dict(version=1, options=self.options, iteration=iteration, rank=rank, world=world,
                     model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                     replay=self.replay.state_dict(), update_count=self.update_count,
                     prediction_rmse_ema=self.prediction_rmse_ema)
        path = Path(directory)/f"ssr_{iteration}_rank_{rank}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def load(self, checkpoint, iteration, rank, world):
        path = Path(checkpoint).parent/f"ssr_{iteration}_rank_{rank}.pt"
        state = torch.load(path, map_location=self.env.device, weights_only=False)
        expected = dict(version=1, options=self.options, iteration=iteration, rank=rank, world=world)
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError("SSR checkpoint schema/configuration/rank mismatch")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.update_count, self.prediction_rmse_ema = state["update_count"], state["prediction_rmse_ema"]
        # Physics/planner resume starts new episodes; unresolved swing labels are invalid.
        self.reset(None)
