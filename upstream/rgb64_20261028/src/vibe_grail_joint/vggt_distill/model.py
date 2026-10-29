"""Shared patch representations; geometry labels and teacher outputs are never inputs."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.modules.cross_attention import CrossAttentionExtractor
from ..velocity.policy import CommandSonicActor


def position_encoding(height, width, channels, device, dtype=torch.float32):
    """Deterministic 2D sine/cosine positions in row-major patch order."""
    if channels % 4:
        raise ValueError('Position channel count must be divisible by four')
    y, x = torch.meshgrid(torch.linspace(0, 1, height, device=device),
                          torch.linspace(0, 1, width, device=device), indexing='ij')
    freq = torch.exp(torch.arange(channels // 4, device=device) *
                     (-math.log(10000.) / max(channels // 4 - 1, 1)))
    x, y = x.flatten()[:, None] * freq * (2 * math.pi), y.flatten()[:, None] * freq * (2 * math.pi)
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), -1).to(dtype)


class SpatialBlock(nn.Module):
    def __init__(self, width=192):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        # Exact identity at initialization preserves the loaded stage-1 actions.
        nn.init.zeros_(self.attention_out.weight); nn.init.zeros_(self.attention_out.bias)
        nn.init.zeros_(self.ff[-1].weight); nn.init.zeros_(self.ff[-1].bias)
        self.width, self.heads = width, 3

    def forward(self, x, positions):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x) + .1 * positions).reshape(b, n, 3, self.heads, d // self.heads).unbind(2)
        attended = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + self.attention_out(attended.transpose(1, 2).reshape(b, n, d))
        return x + self.ff(self.norm2(x))


class SpatialExtractor(CrossAttentionExtractor):
    """The same three residual token blocks are deployed in both experimental arms."""
    def __init__(self, *args, patch_grid=(4, 7), spatial_depth=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_grid = tuple(patch_grid)
        self.spatial_blocks = nn.ModuleList([SpatialBlock(self.token_dim) for _ in range(spatial_depth)])

    def spatial_features(self, tokens):
        if tokens.shape[1] != math.prod(self.patch_grid):
            raise ValueError(f'Expected patch grid {self.patch_grid}, got {tokens.shape}')
        positions = position_encoding(*self.patch_grid, self.token_dim, tokens.device, tokens.dtype)
        layers = []
        for block in self.spatial_blocks:
            tokens = block(tokens, positions)
            layers.append(tokens)
        return layers

    def forward(self, token_td, *query_vecs):
        adapted = token_td.clone(False)
        adapted['img_tokens'] = self.spatial_features(token_td['img_tokens'])[-1]
        return super().forward(adapted, *query_vecs)


class DistilledActor(CommandSonicActor):
    def __init__(self, *args, distillation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation = dict(distillation or {})
        self.alignment_pairs = [(1, 2)] if self.distillation.get('variant') == 'middle' else [(0, 0), (1, 1), (2, 2)]
        if self.distillation.get('variant') not in ('middle', 'hierarchical'):
            raise ValueError('Choose middle or hierarchical distillation')
        # Training-only modules: forward() never calls these heads.
        # Auxiliary head count must not shift the RNG used to initialize the
        # common relation heads or the subsequently constructed PPO critic.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(260026)
            heads = [nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 512), nn.GELU(), nn.Linear(512, 2048))
                     for _ in range(3)]
            self.alignment_heads = nn.ModuleList([heads[level] for level, _ in self.alignment_pairs])
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(260027)
            self.relation_head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 6))
            self.progress_head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 32), nn.ELU(), nn.Linear(32, 1))

    @property
    def spatial_extractor(self):
        candidates = [m for m in self.extractors.values() if isinstance(m, SpatialExtractor)]
        if len(candidates) != 1:
            raise RuntimeError('Exactly one shared visual extractor is required')
        return candidates[0]

    def auxiliary_loss(self, batch):
        obs = batch['obs']
        ext = self.spatial_extractor
        layers = ext.spatial_features(obs['kv_tokens']['img_tokens'])
        terms = []
        for head, (student_level, teacher_level) in zip(self.alignment_heads, self.alignment_pairs):
            prediction = head(layers[student_level])
            target = batch['teacher'][:, teacher_level].detach().float()
            terms.append((1 - F.cosine_similarity(prediction.float(), target, dim=-1, eps=1e-6)).mean())
        spatial = torch.stack(terms).mean()
        # Use the exact action-path extractor, with its proprioceptive/task queries.
        z = self.encode(obs)
        relation = self.relation_head(z)
        valid = batch['relation_valid'].float()
        relation_loss = (F.smooth_l1_loss(relation, batch['relation'], reduction='none') * valid).sum() / valid.sum().clamp_min(1)
        progress = self.progress_head(z).squeeze(-1)
        progress_loss = F.binary_cross_entropy_with_logits(progress, batch['progress'])
        return spatial, relation_loss, progress_loss


def configure_agent(agent, settings):
    agent['actor']['class_name'] = 'vibe_grail_joint.vggt_distill.model.DistilledActor'
    agent['actor']['distillation'] = settings['distillation']
    for cfg in agent['actor']['extractor_cfg'].values():
        cfg['class_name'] = 'vibe_grail_joint.vggt_distill.model.SpatialExtractor'
        cfg['patch_grid'] = [settings['camera']['height'] // 16, settings['camera']['width'] // 16]
        cfg['spatial_depth'] = 3
    agent['algorithm']['class_name'] = 'vibe_grail_joint.vggt_distill.algorithm.DistillationPPO'
    agent['algorithm']['distillation'] = settings['distillation']
    return agent
