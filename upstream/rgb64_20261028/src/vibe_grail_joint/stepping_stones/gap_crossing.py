"""Contact-resolved gap crossing, for reward/evaluation only; no actor inputs."""
import math
import torch

KIND = 'airborne_gap_crossing_v1'
DEFAULTS = dict(kind=KIND, side_force_n=5., support_force_n=20., air_force_n=1.,
                minimum_air_s=.04, minimum_clearance_m=.02,
                stable_landing_s=.04, bridge_failure_s=.02,
                bridge_weight=-2., crossing_weight=.25)


def options(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULTS) or value['kind'] != KIND:
        raise ValueError('Require the complete airborne gap-crossing contract')
    for name, number in value.items():
        if name == 'kind':
            continue
        if not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError('Invalid crossing option '+name)
        if (number >= 0 if name == 'bridge_weight' else number <= 0):
            raise ValueError('Invalid crossing option '+name)
    if not value['air_force_n'] < value['side_force_n'] < value['support_force_n']:
        raise ValueError('Side-load threshold must be below total support threshold')
    return dict(value)


class GapCrossingState:
    """Per-foot/per-gap state. Supports partial single-side stance and skipped tops.

    Inputs: actual contact x/vertical loads, sole extents and edge heights. These
    are privileged supervision, never observations. Two different feet may stand
    on opposite sides; a single foot carrying load on both sides is a bridge.
    """
    def __init__(self, worlds, gaps, device, settings=None):
        self.settings = options(settings or DEFAULTS)
        shape = (worlds, 2, gaps)
        self.last_side = torch.zeros(shape, dtype=torch.int8, device=device)
        self.air = torch.zeros(shape, device=device)
        self.clear = torch.zeros(shape, dtype=torch.bool, device=device)
        self.pending = torch.zeros(shape, dtype=torch.bool, device=device)
        self.landing = torch.zeros(shape, device=device)
        self.crossed = torch.zeros(shape, dtype=torch.bool, device=device)
        self.bridge_time = torch.zeros(shape, device=device)
        self.failed = torch.zeros(worlds, dtype=torch.bool, device=device)
        self.last_bridge = torch.zeros(shape, dtype=torch.bool, device=device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for name in ('last_side', 'air', 'clear', 'pending', 'landing', 'crossed',
                     'bridge_time', 'failed', 'last_bridge'):
            getattr(self, name)[ids] = 0

    @torch.no_grad()
    def update(self, contact_x, vertical_load, sole_min_x, sole_max_x, sole_min_z,
               bounds, edge_height, valid, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Positive physics timestep required')
        p = self.settings
        # Shape: world, foot, contact-slot, gap. Bounds are world-local.
        x = contact_x[..., None]
        load = vertical_load[..., None]
        left = (load*(x <= bounds[:, None, None, :, 0])).sum(2)
        right = (load*(x >= bounds[:, None, None, :, 1])).sum(2)
        active = valid[:, None, :]
        bridge = (left >= p['side_force_n']) & (right >= p['side_force_n']) & active
        self.bridge_time.add_(bridge*dt)
        self.failed |= (self.bridge_time+1e-7 >= p['bridge_failure_s']).any(dim=(1, 2))
        supported = (vertical_load.sum(-1) >= p['support_force_n'])[..., None] & active
        touching = (vertical_load.sum(-1) >= p['air_force_n'])[..., None] & active
        side = torch.where((left >= p['air_force_n']) & (right < p['air_force_n']), -1,
                           torch.where((right >= p['air_force_n']) & (left < p['air_force_n']), 1, 0))
        side = torch.where(touching, side, 0).to(torch.int8)
        # Even light two-sided contact cancels a prospective flight/landing.
        airborne = ~touching & ~bridge & active
        overlaps = ((sole_max_x[..., None] > bounds[:, None, :, 0])
                    & (sole_min_x[..., None] < bounds[:, None, :, 1]))
        clearance = sole_min_z[..., None]-edge_height[:, None, :]
        self.air.add_(airborne*dt)
        self.clear |= airborne & overlaps & (clearance >= p['minimum_clearance_m'])
        touchdown = touching & (self.air > 0)
        valid_flight = (touchdown & (self.last_side == -1) & (side == 1)
                        & (self.air+1e-7 >= p['minimum_air_s']) & self.clear & ~bridge)
        self.pending = torch.where(touchdown, valid_flight, self.pending)
        self.pending &= touching & (side == 1) & ~bridge
        self.landing = torch.where(self.pending & supported, self.landing+dt, 0.)
        event = (self.pending & (self.landing+1e-7 >= p['stable_landing_s'])
                 & ~self.crossed & active & ~self.failed[:, None, None])
        self.crossed |= event
        self.last_side = torch.where(supported & (side != 0), side, self.last_side)
        end_air = touching | bridge | ~active
        self.air.masked_fill_(end_air, 0.)
        self.clear.masked_fill_(end_air, False)
        self.last_bridge.copy_(bridge)
        return bridge.any(-1), event.sum(dim=(1, 2)).float()

    def qualified(self, valid):
        return ((self.crossed | ~valid[:, None, :]).all(dim=(1, 2)) & ~self.failed)
