"""Auxiliary gradients share the PPO/DAgger optimizer and distributed reduction."""
import torch
from ..velocity.dagger_policy import DaggerPPO


class DistillationPPO(DaggerPPO):
    def __init__(self, *args, distillation, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation = dict(distillation)
        self.collector = self.geometry_teacher = None
        self.distillation_updates = 0
        self.gradient_audit = None

    def attach_distillation(self, collector, teacher):
        self.collector, self.geometry_teacher = collector, teacher

    @torch.no_grad()
    def broadcast_parameters(self):
        """Synchronize in place on each rank's device, without CUDA pickling.

        Object broadcasts deserialize rank-zero CUDA tensors onto cuda:0 even
        in other ranks, creating extra contexts and temporary model copies.
        Tensor broadcasts keep parameters and persistent buffers local.
        """
        models = [self._raw_actor, self._raw_critic]
        if self.rnd:
            models.append(self.rnd.predictor)
        for model in models:
            for name, tensor in model.state_dict().items():
                if not torch.is_tensor(tensor) or tensor.device != torch.device(self.device):
                    raise RuntimeError(f'Expected a local model tensor for broadcast: {name}')
                if tensor.numel() == 0:
                    continue
                payload = tensor.contiguous()
                torch.distributed.broadcast(payload, src=0)
                if not tensor.is_contiguous():
                    tensor.copy_(payload)

    def _extra_backward(self):
        super()._extra_backward()
        self._aux_minibatches += 1
        if (self._aux_minibatches - 1) % self.distillation['loss_interval_minibatches']:
            return
        batch = self.collector.sample(self.distillation['batch_size'], self.device)
        spatial, relation, progress = self._raw_actor.auxiliary_loss(batch)
        if self.gradient_audit is None:
            parameters = [p for p in self._raw_actor.spatial_extractor.spatial_blocks.parameters() if p.requires_grad]
            grads = torch.autograd.grad(spatial, parameters, retain_graph=True, allow_unused=True)
            norm = sum(float(g.detach().square().sum()) for g in grads if g is not None) ** .5
            if not norm > 0:
                raise RuntimeError('VGGT loss did not reach shared action-path spatial blocks')
            self.gradient_audit = dict(shared_spatial_gradient_norm=norm, teacher_frozen=True)
        warmup = min(1., (self.distillation_updates + 1) / self.distillation['warmup_updates'])
        weighted = (self.distillation['spatial_coefficient'] * warmup * spatial +
                    self.distillation['relation_coefficient'] * relation +
                    self.distillation['progress_coefficient'] * progress)
        weighted.backward()
        self._aux_count += 1
        for k, v in dict(spatial=spatial, relation=relation, progress=progress, weighted=weighted).items():
            self._aux_sums[k] = self._aux_sums.get(k, 0.) + float(v.detach())

    def update(self):
        self.collector.prepare(self.geometry_teacher)
        self.geometry_teacher.assert_frozen()
        self._aux_sums = {}; self._aux_count = self._aux_minibatches = 0
        losses, info = super().update()
        self.distillation_updates += 1
        losses.update({f'distill_{k}': v / max(1, self._aux_count) for k, v in self._aux_sums.items()})
        info.update({f'Distill/{k}': v for k, v in self.collector.last_metrics.items()})
        info['Distill/shared_gradient_norm'] = self.gradient_audit['shared_spatial_gradient_norm']
        return losses, info

    def save(self):
        state = super().save()
        state['distillation'] = dict(settings=self.distillation, updates=self.distillation_updates,
                                    gradient_audit=self.gradient_audit)
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        saved = loaded_dict.get('distillation', {})
        if saved.get('settings') != self.distillation:
            raise ValueError('Distillation resume settings mismatch')
        result = super().load(loaded_dict, load_cfg, strict)
        self.distillation_updates = saved['updates']; self.gradient_audit = saved['gradient_audit']
        return result
