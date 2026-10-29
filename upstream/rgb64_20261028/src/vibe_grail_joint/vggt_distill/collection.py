"""Collect student-visited RGB/feature pairs and privileged auxiliary targets."""
import time

import torch
from tensordict import TensorDict
from mjlab.utils.lab_api.math import quat_apply
from ..velocity.dagger_env import DaggerVecEnv
from .visual_contract import validate_patch_coverage


class AscentDaggerEnv(DaggerVecEnv):
    @property
    def quality_steps(self): return self.env.quality_steps
    @quality_steps.setter
    def quality_steps(self, value): self.env.quality_steps = value


@torch.no_grad()
def relation_labels(raw, ids, all_edges=False):
    c = raw.command_manager.get_term('motion')
    q = raw.guidance.quality
    tile = c.tile[ids]
    origin = c.tile_origins[tile]
    feet = c.robot_body_pos_w[ids][:, q.feet]
    quaternion = c.robot_body_quat_w[ids][:, q.feet]
    front = feet + quat_apply(quaternion.reshape(-1, 4), feet.new_tensor([.142, 0., -.035]).expand(len(ids) * 2, 3)).reshape(len(ids), 2, 3)
    front = front - origin[:, None]
    edges, jumps = c._profile_edges[tile], c._profile_jumps[tile]
    distance = edges[:, None] - front[..., 0, None]
    edge_mask = jumps[:, None].abs() > 0 if all_edges else jumps[:, None] > 0
    valid = torch.isfinite(edges[:, None]) & edge_mask & (distance >= -.03)
    nearest, index = torch.where(valid, distance, torch.inf).min(-1)
    have = torch.isfinite(nearest)
    tops = c._profile_base[tile, None] + jumps.cumsum(-1)
    top = tops.gather(1, index)
    relative_height = top - front[..., 2]
    local_ids = torch.searchsorted(q.ids, ids)
    if not torch.equal(q.ids[local_ids], ids):
        raise RuntimeError('Quality label environment mapping mismatch')
    supported = q.sensor_support[local_ids].float()
    relation = torch.cat((torch.nan_to_num(nearest, posinf=0.).div(1.5).clamp(-1, 1),
                          relative_height.div(.35).clamp(-2, 2), supported), -1)
    mask = torch.cat((have, have, torch.ones_like(have)), -1)
    root_x = c.robot.data.root_link_pos_w[ids, 0] - origin[:, 0]
    start, goal = c.tile_data['start'][tile], c.tile_data['goal'][tile]
    progress = ((root_x - start) / (goal - start).clamp_min(.1)).clamp(0, 1)
    return relation, mask, progress


class DistillationCollector:
    def __init__(self, wrapped, settings, rank):
        self.env = wrapped
        self.settings = settings
        self.rank = rank
        self.pending = []
        self.replay = None
        self.last_step = -1
        self.rng = torch.Generator(device='cpu').manual_seed(187003 + rank)
        self.total_samples = 0
        self.last_metrics = {}

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def posture_steps(self): return self.env.posture_steps
    @posture_steps.setter
    def posture_steps(self, value): self.env.posture_steps = value
    @property
    def quality_steps(self): return self.env.quality_steps
    @quality_steps.setter
    def quality_steps(self, value): self.env.quality_steps = value

    @torch.no_grad()
    def capture(self, obs):
        step = int(self.unwrapped.common_step_counter)
        if step == self.last_step or step % self.settings['sample_interval_steps']:
            return
        self.last_step = step
        c = self.unwrapped.command_manager.get_term('motion')
        available = c.ppo_online.nonzero().flatten()
        order = torch.randperm(len(available), generator=self.rng)[:self.settings['sample_worlds']].to(available.device)
        ids = available[order]
        rgb = self.unwrapped.scene['head_cam'].data.rgb
        contract = validate_patch_coverage(rgb.shape[1], rgb.shape[2], self.settings['patch_grid'])
        if obs['kv_tokens']['img_tokens'].shape[1] != contract['patch_count']:
            raise RuntimeError('Rendered RGB and actual student token coverage disagree')
        relation, mask, progress = relation_labels(self.unwrapped, ids,
            all_edges=self.settings.get('relation_edge_mode') == 'ascending_and_descending')
        with torch.inference_mode(False):
            data = TensorDict({
                'obs': obs[ids].select('kv_tokens', 'q_cls', 'q_task_cmd', 'q_proprio').detach().to('cpu').clone(),
                'rgb': rgb[ids].to('cpu').clone(),
                'relation': relation.to('cpu').clone(), 'relation_valid': mask.to('cpu').clone(),
                'progress': progress.to('cpu').clone(),
            }, batch_size=[len(ids)])
        self.pending.append(data)

    def get_observations(self):
        obs = self.env.get_observations(); self.capture(obs); return obs

    def reset(self):
        self.pending.clear(); self.last_step = -1
        obs, extras = self.env.reset(); self.capture(obs); return obs, extras

    def step(self, actions):
        obs, reward, done, extras = self.env.step(actions)
        self.capture(obs)
        return obs, reward, done, extras

    def prepare(self, teacher):
        if not self.pending:
            raise RuntimeError('Distillation requires fresh student-visited images')
        started = time.monotonic()
        fresh = torch.cat(self.pending, 0)
        self.pending.clear()
        fresh['teacher'] = teacher(fresh.pop('rgb'))
        self.total_samples += len(fresh)
        self.replay = fresh if self.replay is None else torch.cat((self.replay, fresh), 0)
        # Bounded recent history; teacher and frozen Theia do not change versions.
        self.replay = self.replay[-self.settings['buffer_capacity']:].clone()
        self.last_metrics = dict(teacher_seconds=time.monotonic() - started, fresh_images=len(fresh),
                                 replay_images=len(self.replay), total_images=self.total_samples,
                                 teacher_calls=teacher.calls,
                                 target_spatial_std=float(fresh['teacher'].float().std(dim=2).mean()))

    def sample(self, count, device):
        if self.replay is None:
            raise RuntimeError('No geometry teacher targets prepared')
        ids = torch.randint(len(self.replay), (count,), generator=self.rng)
        return self.replay[ids].to(device).clone()

    def state_dict(self):
        return dict(version=1, rng=self.rng.get_state(), total_samples=self.total_samples,
                    replay=self.replay)

    def load_state_dict(self, state):
        if state['version'] != 1: raise ValueError('Unknown distillation replay version')
        self.rng.set_state(state['rng']); self.total_samples = state['total_samples']; self.replay = state['replay']
