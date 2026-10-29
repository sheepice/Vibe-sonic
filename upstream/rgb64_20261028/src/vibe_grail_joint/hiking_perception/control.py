"""Per-world recurrent state outside PPO; rollout latents remain immutable.

The first controlled experiment freezes the pretrained perception module. This
keeps PPO and the balanced DAgger replay consistent with the same representation.
Perception can subsequently be refit on new-policy trajectories, followed by a
fresh policy continuation; online encoder mutation is deliberately disallowed.
"""
from pathlib import Path
import hashlib
import torch
from .contracts import GeometrySpec, contract, parse_contract
from .models import GeometryMemory


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class PerceptionState:
    def __init__(self, checkpoint, batch, device, chunk=256):
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        geometry, model = parse_contract(state['contract'])
        self.model = GeometryMemory(geometry, model).to(device).eval().requires_grad_(False)
        self.model.load_state_dict(state['encoder'])
        self.h, self.c = self.model.initial(batch, device)
        self.chunk = chunk

    @torch.no_grad()
    def step(self, obs, reset, age):
        for start in range(0, len(self.h), self.chunk):
            sl = slice(start, start+self.chunk)
            h, c = self.model.step(obs['kv_tokens']['img_tokens'][sl], obs['q_proprio'][sl],
                                  age[sl], self.h[sl], self.c[sl], reset[sl])
            self.h[sl], self.c[sl] = h, c
        # Never return the mutable recurrent buffer to rollout/DAgger storage.
        return self.h.clone()


class PerceptionWrapper:
    def __init__(self, env, *, mode, checkpoint=None, chunk=256, oracle_geometry=None):
        if mode not in ('geometry', 'oracle'):
            raise ValueError('Expected geometry or privileged oracle')
        self.env, self.mode = env, mode
        self.state = None
        self._cached = None
        self._reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        if mode == 'geometry':
            if not checkpoint:
                raise ValueError('Geometry policy requires a pretrained perception checkpoint')
            self.state = PerceptionState(checkpoint, env.num_envs, env.device, chunk)
            self.dimension = self.state.model.spec.hidden_dim
            self.metadata = dict(mode=mode, frozen=True, checkpoint_sha256=file_hash(checkpoint),
                                 contract=contract(self.state.model.geometry, self.state.model.spec))
        else:
            from .simulation import GeometryLabels
            geometry = oracle_geometry or GeometrySpec()
            self.labels = GeometryLabels(env.unwrapped, geometry)
            self.dimension = 3*geometry.cells
            self.metadata = dict(mode=mode, privileged=True, deployable=False, contract=contract(geometry))

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _augment(self, obs):
        obs = obs.clone(False)
        if self.mode == 'geometry':
            age = torch.full((self.num_envs,), self.unwrapped.step_dt, device=self.device)
            latent = self.state.step(obs, self._reset, age)
        else:
            c = self.unwrapped.command_manager.get_term('motion')
            ids = c.online.nonzero().flatten()
            labels = self.labels.sample(ids)
            # Replay worlds are flat/GRAIL meshes, not the procedural profiles.
            # Supply unknown zero geometry there instead of incorrect labels.
            latent = torch.zeros(self.num_envs, self.dimension, device=self.device)
            valid = labels['surface_valid']
            latent[ids] = torch.cat((torch.where(valid, labels['height'], 0.),
                                    labels['edge'].float()*valid, valid.float()), -1)
        obs['hiking_latent'] = latent
        self._reset.zero_(); self._cached = obs
        return obs

    def get_observations(self):
        # Repeated reads of one control time must not advance the LSTM.
        return self._cached if self._cached is not None else self._augment(self.env.get_observations())

    def reset(self):
        obs, extras = self.env.reset()
        self._reset.fill_(True)
        return self._augment(obs), extras

    def step(self, actions):
        obs, reward, done, extras = self.env.step(actions)
        self._reset.copy_(done.bool())
        return self._augment(obs), reward, done, extras

    def save(self, path):
        if self.state is not None:
            torch.save(dict(schema=1, mode='geometry', contract=self.metadata['contract'],
                            encoder=self.state.model.state_dict()), str(path)+'.tmp')
            Path(str(path)+'.tmp').replace(path)
