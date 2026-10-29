"""Optional gap rewards, strict route scoring and per-physics-step contact audit."""
import torch

from .gap_crossing import GapCrossingState, options


def configure(cfg, settings, *, rewards=True):
    options(settings['stones']['gap_crossing'])
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    sensors = list(cfg.scene.sensors)
    for side in ('left', 'right'):
        sensors.append(ContactSensorCfg(name=f'gap_contact_{side}',
            primary=ContactMatch(mode='geom', pattern=f'{side}_foot[1-7]_collision', entity='robot'),
            secondary=ContactMatch(mode='body', pattern='terrain'),
            fields=('found', 'force', 'normal', 'tangent', 'pos'), reduce='maxforce', num_slots=4,
            global_frame=True))
    cfg.scene.sensors = tuple(sensors)
    if rewards:
        value = settings['stones']['gap_crossing']
        # This framework observes the reward-term vector in the critic. Keep its
        # dimension/order so the full critic and optimizer can be restored. Log
        # the new components separately while composing the existing two slots.
        low, safe = cfg.rewards['stone_low_support'], cfg.rewards['stone_safe_support']
        if not low.weight < 0 < safe.weight:
            raise ValueError('Expected original sparse support reward signs')
        low.func = low_support
        low.params = {'bridge_scale':value['bridge_weight']/low.weight}
        cfg.rewards['stone_safe_support'].func = safe_support
        cfg.rewards['stone_safe_support'].params = {'crossing_scale':value['crossing_weight']/safe.weight}
        cfg.terminations['online_success'].func = course_success
        cfg.terminations['online_success'].params = {}
    return cfg


class GapCrossingObserver:
    def __init__(self, env, settings):
        self.env = env
        self.c = c = env.command_manager.get_term('motion')
        self.settings = options(settings['stones']['gap_crossing'])
        tiles = c.cfg.random_tiles
        gaps = max(1, max(len(t.get('gaps', [])) for t in tiles))
        self.bounds = torch.zeros(len(tiles), gaps, 2, device=env.device)
        self.heights = torch.zeros(len(tiles), gaps, device=env.device)
        self.exists = torch.zeros(len(tiles), gaps, dtype=torch.bool, device=env.device)
        for i, tile in enumerate(tiles):
            for j, (a, b) in enumerate(tile.get('gaps', [])):
                edges = [s['top'] for s in tile['solids']
                         if abs(s['x']+s['hx']-a) < 1e-5 or abs(s['x']-s['hx']-b) < 1e-5]
                if len(edges) < 2:
                    raise ValueError('Missing physical surfaces on both sides of a gap')
                self.bounds[i, j] = torch.tensor([a, b], device=env.device)
                self.heights[i, j] = max(edges)
                self.exists[i, j] = True
        self.state = GapCrossingState(env.num_envs, gaps, env.device, self.settings)
        self.feet = [list(c.cfg.body_names).index(f'{s}_ankle_roll_link') for s in ('left', 'right')]
        self.corners = torch.tensor([[-.064,-.037,-.035], [-.064,.037,-.035],
                                     [.142,-.037,-.035], [.142,.037,-.035]], device=env.device)
        self.bridge_seconds = torch.zeros(env.num_envs, device=env.device)
        self.crossings = torch.zeros_like(self.bridge_seconds)
        self.valid = torch.zeros(env.num_envs, gaps, dtype=torch.bool, device=env.device)

    def begin_step(self):
        self.bridge_seconds.zero_(); self.crossings.zero_()

    def reset(self, ids=None):
        self.state.reset(ids)
        self.valid[slice(None) if ids is None else ids] = False

    @torch.no_grad()
    def observe_substep(self):
        from mjlab.utils.lab_api.math import quat_apply
        c = self.c
        active = c.ppo_online & (c.family[c.tile] > 0)
        self.valid = self.exists[c.tile] & active[:, None]
        origin = c.tile_origins[c.tile]
        xs, loads = [], []
        for side in ('left', 'right'):
            data = self.env.scene[f'gap_contact_{side}'].data
            pos = data.pos.reshape(self.env.num_envs, -1, 3)
            force = data.force.reshape_as(pos)
            normal = data.normal.reshape_as(pos)
            found = data.found.reshape(self.env.num_envs, -1) > 0
            weight = torch.where(found & (normal[..., 2].abs() > .7), force[..., 2].abs(), 0.)
            xs.append(pos[..., 0]-origin[:, None, 0]); loads.append(weight)
        position = c.robot_body_pos_w[:, self.feet]
        quat = c.robot_body_quat_w[:, self.feet, None].expand(-1,-1,4,-1)
        corners = position[:, :, None]+quat_apply(quat.reshape(-1,4),
                    self.corners.expand(self.env.num_envs,2,4,3).reshape(-1,3)).reshape(self.env.num_envs,2,4,3)
        corners = corners-origin[:, None, None, :]
        bridge, events = self.state.update(torch.stack(xs, 1), torch.stack(loads, 1),
            corners[...,0].amin(-1), corners[...,0].amax(-1), corners[...,2].amin(-1),
            self.bounds[c.tile], self.heights[c.tile], self.valid, self.env.physics_dt)
        self.bridge_seconds.add_(bridge.any(-1)*self.env.physics_dt)
        self.crossings.add_(events)

    @property
    def qualified(self):
        return self.state.qualified(self.valid)

    def row(self, i):
        return dict(gap_crossing_qualified=bool(self.qualified[i]),
                    same_foot_bridge_failed=bool(self.state.failed[i]),
                    max_same_foot_bridge_seconds=float(self.state.bridge_time[i].max()),
                    required_gaps=int(self.valid[i].sum()),
                    crossed_by_foot=self.state.crossed[i, :, self.valid[i]].tolist())


def install(env, settings):
    if hasattr(env, 'gap_crossing'):
        raise ValueError('Gap observer already installed')
    observer = env.gap_crossing = GapCrossingObserver(env, settings)
    previous_substep = env.metrics_manager.compute_substep
    def substep():
        previous_substep(); observer.observe_substep()
    env.metrics_manager.compute_substep = substep
    previous_step = env.step
    def step(actions):
        observer.begin_step()
        return previous_step(actions)
    env.step = step
    previous_reset = env._reset_idx
    def reset(ids=None):
        observer.reset(ids)
        return previous_reset(ids)
    env._reset_idx = reset
    return observer


def bridge_cost(env):
    observer = env.gap_crossing
    active = observer.valid.any(-1)
    denominator = active.sum().clamp_min(1)
    env.extras.setdefault('log', {}).update({
        'Stones/GapCrossing/bridge_world_fraction': ((observer.bridge_seconds > 0) & active).sum()/denominator,
        'Stones/GapCrossing/failed_episode_fraction': (observer.state.failed & active).sum()/denominator,
        'Stones/GapCrossing/valid_crossings': observer.crossings.sum(),
        'Stones/GapCrossing/qualified_world_fraction': (observer.qualified & active).sum()/denominator})
    return observer.bridge_seconds/env.step_dt


def crossing_reward(env):
    # RewardManager multiplies by control dt; each valid landing earns one event.
    return env.gap_crossing.crossings/env.step_dt


def low_support(env, bridge_scale):
    return env.guidance.reward('low_support')+bridge_scale*bridge_cost(env)


def safe_support(env, crossing_scale=0.):
    value = env.guidance.reward('safe_support')
    return torch.where(env.gap_crossing.bridge_seconds > 0, 0., value)+crossing_scale*crossing_reward(env)


def course_success(env):
    from .start_stop_task import course_and_stop_success
    c = env.command_manager.get_term('motion')
    result = course_and_stop_success(env)
    c.terminal_success = result & env.gap_crossing.qualified
    return c.terminal_success
