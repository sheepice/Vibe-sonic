"""SSR Section 3.2: support deficiency, Gaussian contact prediction, delayed labels.

These are training-only utilities. They do not modify the actor or its inputs.
Paper: https://arxiv.org/html/2605.30770v1#S3.SS2
"""

import math
import torch
from torch import nn


def rotate_xy(xy, yaw):
    co, si = yaw.cos(), yaw.sin()
    while co.ndim < xy.ndim-1:
        co, si = co.unsqueeze(-1), si.unsqueeze(-1)
    return torch.stack((co*xy[..., 0]-si*xy[..., 1],
                        si*xy[..., 0]+co*xy[..., 1]), -1)


def yaw_angle(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def patch_offsets(length=.206, width=.074, spacing=.025, *, device="cpu"):
    # Include both boundaries; actual spacing never exceeds the requested value.
    x = torch.linspace(-length/2, length/2, math.ceil(length/spacing)+1, device=device)
    y = torch.linspace(-width/2, width/2, math.ceil(width/spacing)+1, device=device)
    return torch.cartesian_prod(x, y)


def support_deficiency(heights, sole_height=None, tolerance=.025):
    """SSR Eq. 2 (one-sided height test, not the legacy soft support threshold).

    Imagined contacts use the patch maximum. A stance contact uses the actual
    sole-center height. Obstacles ABOVE that height are not rejected by Eq. 2;
    this term is therefore not a replacement for general collision detection.
    """
    if sole_height is None:
        sole_height = heights.amax(-1)
    return 1-((sole_height[..., None]-heights) < tolerance).float().mean(-1)


def gaussian_quadrature(mu, sigma):
    """Nine deterministic Gaussian samples with matching first/second moments."""
    nodes = mu.new_tensor([-math.sqrt(3), 0., math.sqrt(3)])
    grid = torch.cartesian_prod(nodes, nodes)
    weights = mu.new_tensor([1/6, 2/3, 1/6])
    weights = (weights[:, None]*weights[None, :]).flatten()
    return mu.unsqueeze(-2)+sigma[..., None, None]*grid, weights


def foothold_score(stance, stance_deficit, swing_deficit, swing_enabled, variance=.0625):
    """SSR Eq. 3. Warm-up uses stance assessment until predictions are usable."""
    deficiency = torch.where(stance, stance_deficit, swing_deficit*swing_enabled)
    return torch.exp(-deficiency.sum(-1).square()/variance)


class ContactPredictor(nn.Module):
    """Shared [256,128] ELU MLP; each foot emits two means and one isotropic sigma."""
    def __init__(self, input_dim=228, sigma_min=.015, sigma_max=.30):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 6))
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        values = self.net(features).reshape(-1, 2, 3)
        mu = values[..., :2]
        sigma = self.sigma_min+(self.sigma_max-self.sigma_min)*values[..., 2].sigmoid()
        return mu, sigma


def gaussian_nll(mu, sigma, target):
    """Negative log likelihood in 2D, omitting the constant log(2*pi)."""
    return ((target-mu).square().sum(-1)/(2*sigma.square())+2*sigma.log()).mean()


class DelayedContacts:
    """Bounded swing histories, labelled only by the first subsequent valid contact.

    Every sample retains its OWN base frame. Invalid episodes, expired swing
    samples and reset states never create labels. No target is a future planner
    reference or a contact from a new episode.
    """
    def __init__(self, envs, feature_dim, slots, max_age_steps, device):
        self.features = torch.zeros(envs, 2, slots, feature_dim, device=device)
        self.origins = torch.zeros(envs, 2, slots, 2, device=device)
        self.yaws = torch.zeros(envs, 2, slots, device=device)
        self.steps = torch.full((envs, 2, slots), -max_age_steps-1, device=device, dtype=torch.long)
        self.valid = torch.zeros(envs, 2, slots, device=device, dtype=torch.bool)
        self.max_age_steps, self.slots, self.cursor = max_age_steps, slots, 0

    def add(self, features, origins, yaw, swing, step):
        k = self.cursor % self.slots
        self.features[:, :, k] = features[:, None]
        self.origins[:, :, k] = origins[:, None]
        self.yaws[:, :, k] = yaw[:, None]
        self.steps[:, :, k] = step
        self.valid[:, :, k] = swing
        self.cursor += 1

    def reset(self, mask):
        self.valid[mask] = False

    def resolve(self, contacts, landing_xy, terminated, step):
        self.valid[terminated] = False
        self.valid &= (step-self.steps <= self.max_age_steps) & (step-self.steps > 0)
        selected = self.valid & contacts[..., None]
        ids, feet, slots = selected.nonzero(as_tuple=True)
        target = rotate_xy(landing_xy[ids, feet]-self.origins[ids, feet, slots],
                           -self.yaws[ids, feet, slots])
        features = self.features[ids, feet, slots].clone()
        self.valid &= ~contacts[..., None]
        return features, feet, target


class ContactReplay:
    """Fixed GPU FIFO of detached, fully labelled training examples."""
    def __init__(self, capacity, feature_dim, device, seed):
        self.features = torch.zeros(capacity, feature_dim, device=device)
        self.feet = torch.zeros(capacity, device=device, dtype=torch.long)
        self.targets = torch.zeros(capacity, 2, device=device)
        self.capacity, self.size, self.cursor, self.seen = capacity, 0, 0, 0
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def add(self, features, feet, targets):
        if not len(features):
            return
        if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
            raise ValueError("Non-finite SSR contact training example")
        self.seen += len(features)
        features, feet, targets = features[-self.capacity:], feet[-self.capacity:], targets[-self.capacity:]
        ids = (torch.arange(len(features), device=features.device)+self.cursor) % self.capacity
        self.features[ids], self.feet[ids], self.targets[ids] = features, feet, targets
        self.cursor = (self.cursor+len(features)) % self.capacity
        self.size = min(self.capacity, self.size+len(features))

    def sample(self, count):
        if not self.size:
            raise ValueError("Cannot sample empty contact replay")
        ids = torch.randint(self.size, (count,), generator=self.generator, device=self.features.device)
        # Clone outside inference_mode so backward can save inputs safely.
        return self.features[ids].clone(), self.feet[ids].clone(), self.targets[ids].clone()

    def state_dict(self):
        return dict(features=self.features[:self.size].cpu(), feet=self.feet[:self.size].cpu(),
                    targets=self.targets[:self.size].cpu(), size=self.size, cursor=self.cursor,
                    seen=self.seen, capacity=self.capacity, rng=self.generator.get_state())

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity or state["features"].shape[1:] != self.features.shape[1:]:
            raise ValueError("SSR contact replay layout mismatch")
        size = int(state["size"])
        if not 0 <= size <= self.capacity or not 0 <= state["cursor"] < self.capacity:
            raise ValueError("Invalid SSR contact replay indices")
        for name in ("features", "feet", "targets"):
            value = state[name]
            if len(value) != size or not torch.isfinite(value).all():
                raise ValueError("Invalid SSR contact replay values")
            getattr(self, name)[:size] = value.to(self.features.device)
        self.size, self.cursor, self.seen = size, state["cursor"], state["seen"]
        self.generator.set_state(state["rng"].cpu())
