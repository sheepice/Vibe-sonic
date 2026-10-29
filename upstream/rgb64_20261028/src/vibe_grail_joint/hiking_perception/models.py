"""GaussGym-inspired recurrent terrain representation and local dynamics model."""
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import GeometrySpec, ModelSpec
from .geometry import height_bins, decode_height, masked_mean


class GeometryMemory(nn.Module):
    """Frozen backbone tokens in; trainable spatial fusion, memory and 3-D head."""
    def __init__(self, geometry=GeometrySpec(), model=ModelSpec()):
        super().__init__(); self.geometry, self.spec = geometry, model
        h = model.hidden_dim
        self.patch = nn.Sequential(nn.LayerNorm(model.token_dim), nn.Linear(model.token_dim, 24), nn.ELU())
        self.visual = nn.Sequential(nn.Linear(model.tokens*24, h), nn.LayerNorm(h), nn.ELU())
        self.proprio = nn.LayerNorm(model.proprio_dim)
        self.fuse = nn.Sequential(nn.Linear(h+model.proprio_dim+1, h), nn.ELU())
        self.memory = nn.LSTMCell(h, h)
        self.seed = nn.Linear(h, 16*2*2*3)
        self.decoder = nn.Sequential(nn.ConvTranspose3d(16, 12, 4, 2, 1), nn.ELU(),
                                     nn.ConvTranspose3d(12, 8, 4, 2, 1), nn.ELU(),
                                     nn.Conv3d(8, 2, 3, padding=1))
        self.edge = nn.Linear(h, geometry.cells)

    def initial(self, batch, device=None):
        return torch.zeros(batch, self.spec.hidden_dim, device=device), torch.zeros(batch, self.spec.hidden_dim, device=device)

    def step(self, tokens, proprio, age, hidden, cell, reset):
        keep = (~reset.bool()).to(hidden.dtype).unsqueeze(-1)
        if not self.spec.temporal:
            keep = torch.zeros_like(keep)
        visual = self.visual(self.patch(tokens).flatten(1))
        fused = self.fuse(torch.cat((visual, self.proprio(proprio), age.reshape(-1, 1)), -1))
        return self.memory(fused, (hidden*keep, cell*keep))

    def heads(self, hidden):
        grid = self.decoder(self.seed(hidden).reshape(-1, 16, 2, 2, 3))
        grid = F.interpolate(grid, size=(self.geometry.nx, self.geometry.ny, self.geometry.height_bins),
                             mode='trilinear', align_corners=False)
        return grid[:, 0].flatten(1, 2), grid[:, 1].flatten(1, 2), self.edge(hidden)

    def sequence(self, tokens, proprio, age, reset, initial=None):
        h, c = self.initial(tokens.shape[1], tokens.device) if initial is None else initial
        outputs = []
        for t in range(len(tokens)):
            h, c = self.step(tokens[t], proprio[t], age[t], h, c, reset[t])
            outputs.append(h)
        return torch.stack(outputs), (h, c)


def geometry_loss(model, hidden, target):
    logits, offset, edge = model.heads(hidden)
    # Invalid targets are sanitized before computing BCE/MSE.
    heights = torch.nan_to_num(target['height'])
    index, fraction, bounded = height_bins(heights, model.geometry)
    mask = target['visible'].bool() & bounded & torch.isfinite(target['height'])
    onehot = F.one_hot(index, model.geometry.height_bins).float()
    bce = F.binary_cross_entropy_with_logits(logits, onehot, reduction='none',
        pos_weight=logits.new_tensor(float(model.geometry.height_bins-1))).mean(-1)
    selected = offset.gather(-1, index[..., None]).squeeze(-1).sigmoid()
    mse = (selected-fraction).square()
    edge_loss = F.binary_cross_entropy_with_logits(edge, target['edge'].float(), reduction='none',
                                                    pos_weight=edge.new_tensor(4.))
    total = masked_mean(bce, mask) + masked_mean(mse, mask) + .25*masked_mean(edge_loss, mask)
    with torch.no_grad():
        mae = masked_mean((decode_height(logits, offset, model.geometry)-heights).abs(), mask)
        metrics = {'geometry_loss': float(total), 'height_mae_m': float(mae),
                   'visible_cells': int(mask.sum()), 'edge_bce': float(masked_mean(edge_loss, mask))}
    return total, metrics


class ActionWorldModel(nn.Module):
    """Action-conditioned short rollouts. Predict labels, not synthetic RGB."""
    def __init__(self, model=ModelSpec()):
        super().__init__(); self.spec = model
        h = model.hidden_dim
        self.dynamics = nn.GRUCell(model.action_dim + 8 + 6, h)
        self.state = nn.Linear(h, model.future_state_dim)
        self.contact = nn.Linear(h, 4)  # left/right foot-riser and left/right shin

    def forward(self, latent, actions, commands, reference_twist):
        states, contacts = [], []
        for t in range(actions.shape[0]):
            latent = self.dynamics(torch.cat((actions[t], commands[t], reference_twist[t]), -1), latent)
            states.append(self.state(latent)); contacts.append(self.contact(latent))
        return torch.stack(states), torch.stack(contacts)


def future_mask(done, horizons):
    """done[t] ends transition t→t+1. No prediction may cross that reset."""
    alive = ~done.bool().cummax(0).values
    return torch.stack([alive[k-1] for k in horizons])


def world_loss(model, latent, batch):
    state, contact = model(latent, batch['action'], batch['command'], batch['reference_twist'])
    steps = [k-1 for k in model.spec.horizon_steps]
    mask = future_mask(batch['done'], model.spec.horizon_steps)
    # Future states are measured at t+1; contact labels integrate the preceding transition.
    state_loss = (state[steps]-batch['next_state'][steps]).square().mean(-1)
    # Risk is whether ANY contact occurred up to the requested horizon.
    labels = batch['contact'].cummax(0).values[steps]
    pred_risk = torch.stack([contact[:k].amax(0) for k in model.spec.horizon_steps])
    bce = F.binary_cross_entropy_with_logits(pred_risk, labels, reduction='none',
                                            pos_weight=state.new_tensor(5.)).mean(-1)
    total = masked_mean(state_loss, mask)+masked_mean(bce, mask)
    return total, {'world_loss': float(total.detach()), 'valid_horizons': int(mask.sum()),
                   'future_state_mse': float(masked_mean(state_loss, mask).detach())}
