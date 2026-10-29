"""Separate geometry optimizer, frame microbatches retain full sequence gradients."""
from contextlib import nullcontext
import torch
from .model import reconstruction_loss


def autocast(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()


def supervised_update(model,batch,optimizer,burn_in=16,stride=8,microbatch=2,initial=None):
    model.train();optimizer.zero_grad(set_to_none=True)
    # LSTM is FP32. Memory gradients span the entire sequence including burn-in.
    latent,_=model.sequence(batch['tokens'],batch['proprio'],batch['reset'],initial)
    latent=latent[burn_in::stride].flatten(0,1)
    targets={k:v[burn_in::stride].flatten(0,1) for k,v in batch.items() if k not in ('tokens','proprio','reset')}
    count=len(latent);total=0.;metrics={}
    if not count:raise ValueError('No supervised time points')
    for start in range(0,count,microbatch):
        end=min(start+microbatch,count);weight=(end-start)/count
        with autocast(latent.device):
            loss,values=reconstruction_loss(model,latent[start:end],{k:v[start:end] for k,v in targets.items()})
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite reconstruction loss')
        (loss*weight).backward(retain_graph=end<count)
        total+=float(loss.detach())*weight
        for k,v in values.items():metrics[k]=metrics.get(k,0.)+weight*v
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();model.eval()
    metrics.update(loss=total,gradient_norm=float(norm))
    return metrics
