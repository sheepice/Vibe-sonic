"""Scene assembly and privileged labels, never imported by deployment runtime."""
import copy
from dataclasses import fields, replace
import torch
from .contracts import GeometrySpec, require_hiking, scene_key
from .geometry import grid_world, height_labels, project, visible_from_depth, rotate_yaw, yaw_from_quat


def add_depth(cfg):
    cfg.scene.sensors = tuple(replace(s, data_types=tuple(dict.fromkeys((*s.data_types, 'depth'))))
        if s.name == 'head_cam' else s for s in cfg.scene.sensors)
    return cfg


def evaluation_env(settings, num_envs=40, seed=180017, uniform=False, duration=30., record_case=None, approach_extra=0., domain_randomization=False, collection_cases=False):
    import mjlab
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from orcs.core.robustness import strip_domain
    from ..velocity.quality_task import env_config
    from ..velocity.eval_profile_command import ProfileEvaluationCommandCfg
    from ..velocity.step_quality_observer import StepQualityObserver, tile_profile
    from ..velocity.uniform_probe20 import probe_tiles
    require_hiking(settings)
    settings = copy.deepcopy(settings); settings['seed'] = seed
    bank = settings['bank_root']+'/rank_0'
    cfg = env_config(bank, settings, max(20, ((num_envs+19)//20)*20))
    cfg.scene.num_envs = num_envs
    if uniform:
        tiles = []
        for tile in probe_tiles():
            tile = dict(tile); edges, jumps, base = tile_profile(tile)
            tile.update(edges=edges, jumps=jumps, base=base, treads=[tile['tread']]*max(1,len(edges)))
            tiles.append(tile)
        cfg.commands['motion'].random_tiles = tuple(tiles)
        cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs'].random_tiles = tuple(tiles)
    tiles = cfg.commands['motion'].random_tiles
    if approach_extra:
        if not uniform or not 0 <= approach_extra <= 2.:
            raise ValueError('Additional flat approach must be 0..2 m in the uniform probe')
        tiles = tuple(dict(t, edges=[e+approach_extra for e in t['edges']], start=t['start']+approach_extra,
                           goal=t['goal']+approach_extra) if t['edges'] else dict(t) for t in tiles)
        cfg.commands['motion'].random_tiles = tiles
        cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs'].random_tiles = tiles
    speeds = [.25, .4, .7]
    import random
    sampling_rng = random.Random(seed+73)
    cases = []
    for i in range(48 if record_case is not None else num_envs):
        tile = i % len(tiles) if not uniform else (i//12) % 4
        speed = speeds[(i//4) % 3] if uniform else speeds[(i//len(tiles)) % 3]
        phase = [0, 5, 10, 15][i % 4]
        if collection_cases:
            # A small collection still covers all speeds; phase and lateral
            # spawn must not be deterministically tied to terrain kind.
            speed = speeds[(i//len(tiles)+i)%3]
            phase = sampling_rng.choice([0,5,10,15])
        spawn = [sampling_rng.uniform(-.15,.15),sampling_rng.uniform(-.15,.15)] if collection_cases else [-.15+.02*phase,-.15+.02*phase]
        cases.append(dict(id=i, tile=tile, speed=speed, stream=settings['planner_speeds'].index(speed),
                          phase=phase, spawn_xy=spawn))
    if record_case is not None:
        if not uniform or not 0 <= record_case < 48:
            raise ValueError('Recording requires uniform case 0..47')
        cases = [cases[record_case]]
        cfg.scene.num_envs = 1
        add_review_camera(cfg, tiles[cases[0]['tile']])
    old = cfg.commands['motion']
    params = {f.name: getattr(old, f.name) for f in fields(ProfileEvaluationCommandCfg) if hasattr(old, f.name)}
    params.update(cases=tuple(cases), pose_range={}, velocity_range={}, joint_position_range=(0., 0.))
    cfg.commands['motion'] = ProfileEvaluationCommandCfg(**params)
    if not domain_randomization:
        strip_domain(cfg)
    cfg.events.pop('policy_update_counter', None)
    if not domain_randomization:
        for group in cfg.observations.values():
            group.enable_corruption = False
            for term in group.terms.values():
                term.noise = None
    cfg.terminations.clear(); cfg.rewards = {}; cfg.episode_length_s = duration+5
    cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs'].rows = ()
    add_depth(cfg)

    class ObservedEnv(ManagerBasedRlEnv):
        def __init__(self, cfg, **kwargs):
            super().__init__(cfg, **kwargs)
            self.quality = StepQualityObserver(self)
            old_substep = self.metrics_manager.compute_substep
            def substep():
                old_substep(); self.quality.observe_substep()
            self.metrics_manager.compute_substep = substep

        def step(self, actions):
            self.quality.begin_step()
            return super().step(actions)

        def _reset_idx(self, ids=None):
            if hasattr(self, 'quality'):
                self.quality.reset(ids)
            return super()._reset_idx(ids)

    env = RslRlVecEnvWrapper(ObservedEnv(cfg, device='cuda:0'))
    return env, settings, cases


def add_review_camera(cfg, tile):
    """World-fixed side camera encompassing only the robot and selected stairs."""
    import itertools
    import numpy as np
    from scipy.spatial.transform import Rotation
    from mjlab.sensor import CameraSensorCfg
    from ..velocity.step_quality_observer import tile_profile
    _, jumps, base = tile_profile(tile)
    height = max([base, *list(base+np.cumsum(jumps))])
    low, high = np.array([-.5,-.7,-.1]), np.array([tile['goal']+.6,.7,height+1.2])
    center=(low+high)/2
    z=np.array([-.15,-1.,.35]); z/=np.linalg.norm(z)
    x=np.cross([0.,0.,1.],z); x/=np.linalg.norm(x); y=np.cross(z,x)
    rotation=np.stack((x,y,z),axis=1)
    projected=(np.array(list(itertools.product(*zip(low,high))))-center)@rotation
    tangent=np.tan(np.deg2rad(43./2))
    distance=max(np.max(abs(projected[:,0])/(tangent*640/360)+projected[:,2]),
                 np.max(abs(projected[:,1])/tangent+projected[:,2]))*1.04
    size=cfg.scene.terrain.terrain_generator.size
    origin=np.array(tile['origin'])+[-size[0]/2,-size[1]/2,0.]
    cfg.scene.sensors=(*cfg.scene.sensors,CameraSensorCfg(name='review_camera',parent_body=None,
        pos=tuple(origin+center+distance*z),quat=tuple(Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]),
        width=640,height=360,fovy=43.,data_types=('rgb',),enabled_geom_groups=(0,2),use_shadows=False,use_textures=True))


def load_baseline(settings, obs, checkpoint, device):
    from ..velocity.config import agent_config
    from rsl_rl.utils import resolve_callable
    cfg = agent_config(settings['bank_root']+'/rank_0', settings, 20)
    acfg = dict(cfg['actor']); cls = resolve_callable(acfg.pop('class_name'))
    actor = cls(obs=obs, obs_groups=cfg['obs_groups'], obs_set='actor', output_dim=29, **acfg).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    missing, unexpected = actor.load_state_dict(state['actor_state_dict'], strict=False)
    expected = {'command_encoder.0.weight', 'command_encoder.0.bias', 'command_encoder.2.weight', 'command_encoder.2.bias'}
    if unexpected or (missing and set(missing) != expected):
        raise ValueError(f'Not an original stage-1 or velocity baseline: {missing}, {unexpected}')
    return actor


class GeometryLabels:
    def __init__(self, env, spec=GeometrySpec(), history_frames=0, history_stride=5):
        from collections import deque
        from ..velocity.step_quality_observer import tile_profile
        self.env, self.spec = env, spec
        self.history = deque(maxlen=history_frames)
        self.history_stride = history_stride
        self.last_history_step = -1
        self.epoch = torch.zeros(env.num_envs,dtype=torch.long,device=env.device)

        self.c = env.command_manager.get_term('motion')
        profiles = [tile_profile(t) for t in self.c.cfg.random_tiles]
        size = max(1, max(len(p[0]) for p in profiles))
        self.edges = torch.full((len(profiles), size), float('inf'), device=env.device)
        self.jumps = torch.zeros_like(self.edges)
        self.base = torch.tensor([p[2] for p in profiles], device=env.device)
        for i, (edges, jumps, _) in enumerate(profiles):
            self.edges[i, :len(edges)] = self.edges.new_tensor(edges)
            self.jumps[i, :len(jumps)] = self.jumps.new_tensor(jumps)
        self.keys = [scene_key(t) for t in self.c.cfg.random_tiles]
        self.scene_ids = torch.tensor([int(k[:15], 16) for k in self.keys], device=env.device)
        names = list(self.c.cfg.body_names)
        self.feet = [names.index(f'{s}_ankle_roll_link') for s in ('left', 'right')]
        # Knee links are the physical shins in the G1 articulation.
        robot_names = list(env.scene['robot'].body_names)
        self.shins = [robot_names.index(f'{s}_knee_link') for s in ('left', 'right')]

    def reset(self, ids):
        self.epoch[ids] += 1

    @torch.no_grad()
    def sample(self, ids):
        env, c = self.env, self.c
        root = c.robot.data.root_link_pos_w[ids]
        yaw = yaw_from_quat(c.robot.data.root_link_quat_w[ids])
        tile = c.tile[ids]
        world = grid_world(root, yaw, self.spec)
        labels = height_labels(world, root, c.tile_origins[tile], self.edges[tile], self.jumps[tile], self.base[tile], self.spec)
        sensor = env.scene['head_cam']; ci = sensor.camera_idx
        uv, depth, fov = project(labels['points_w'], env.sim.data.cam_xpos[ids, ci],
            env.sim.data.cam_xmat[ids, ci].reshape(-1, 3, 3), sensor.cfg.width, sensor.cfg.height, sensor.cfg.fovy)
        labels['visible'] = labels['surface_valid'] & visible_from_depth(uv, depth, fov,
            sensor.data.depth[ids], self.spec.visibility_tolerance)
        historical = torch.zeros_like(labels['visible'])
        seen_at = torch.full_like(labels['height'],-1.)
        for past in self.history:
            if not torch.equal(past['ids'],ids): raise ValueError('History labeler requires a fixed world subset')
            puv,pdepth,pfov=project(labels['points_w'],past['position'],past['rotation'],
                sensor.cfg.width,sensor.cfg.height,sensor.cfg.fovy)
            same=(past['epoch']==self.epoch[ids]) & (past['tile']==tile)
            seen=visible_from_depth(puv,pdepth,pfov,past['depth'],self.spec.visibility_tolerance) & same[:,None]
            historical |= seen
            seen_at=torch.where(seen,float(past['step'])*env.step_dt,seen_at)
        labels['historical_visible'] = historical & labels['surface_valid'] & ~labels['visible']
        labels['unknown'] = labels['surface_valid'] & ~(labels['visible'] | historical)
        labels['historical_seen_at_s'] = seen_at
        step=int(env.common_step_counter)
        if self.history.maxlen and step%self.history_stride==0 and step!=self.last_history_step:
            self.history.append(dict(ids=ids.clone(),position=env.sim.data.cam_xpos[ids,ci].clone(),
                rotation=env.sim.data.cam_xmat[ids,ci].reshape(-1,3,3).clone(),depth=sensor.data.depth[ids].clone(),
                epoch=self.epoch[ids].clone(),tile=tile.clone(),step=step))
            self.last_history_step=step
        labels['uv'] = uv; labels['camera_depth'] = depth
        labels['scene_id'] = self.scene_ids[tile]
        feet = rotate_yaw(c.robot_body_pos_w[ids][:, self.feet]-root[:, None], -yaw).flatten(1)
        shins = rotate_yaw(env.scene['robot'].data.body_link_pos_w[ids][:, self.shins]-root[:, None], -yaw).flatten(1)
        vel = rotate_yaw(c.robot.data.root_link_lin_vel_w[ids], -yaw)
        labels['state'] = torch.cat((feet, shins, vel), -1)
        return labels

    def contacts(self, ids):
        raw = self.env
        quality = raw.quality if hasattr(raw, 'quality') else raw.guidance.quality
        # A guidance observer may only cover PPO-online worlds. Map explicitly.
        lookup = torch.full((raw.num_envs,), -1, device=raw.device, dtype=torch.long)
        lookup[quality.ids] = torch.arange(len(quality.ids), device=raw.device)
        selected = lookup[ids]
        if (selected < 0).any():
            raise ValueError('Contact labels requested outside instrumented worlds')
        d = quality.step_data
        return torch.cat((d['foot_riser_force_n'][selected] > 15., d['shin_force_n'][selected] > 15.), -1).float()
