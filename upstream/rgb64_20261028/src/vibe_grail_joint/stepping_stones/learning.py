"""Separate geometry optimizer, frame microbatches retain full sequence gradients."""
from contextlib import nullcontext
import torch
from .loss import reconstruction_loss


def autocast(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()


def supervised_update(model,batch,optimizer,burn_in=16,stride=8,microbatch=2,initial=None,*,occupancy_objective=None):
    loss_function=reconstruction_loss
    loss_options={}
    if occupancy_objective is not None:
        from .balanced_loss import KIND,reconstruction_loss as balanced_reconstruction_loss
        loss_options=dict(occupancy_objective)
        kind=loss_options.pop('kind',None)
        if kind==KIND:
            allowed={'scale','region_mix'}
            loss_function=balanced_reconstruction_loss
        else:
            from .front_geometry_loss import KIND as FRONT_KIND, OPTIONS, reconstruction_loss as front_loss
            if kind!=FRONT_KIND:raise ValueError('Unknown geometry occupancy objective')
            allowed=OPTIONS
            loss_function=front_loss
        if set(loss_options)-allowed:raise ValueError('Unknown geometry objective option')
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
            loss,values=loss_function(model,latent[start:end],{k:v[start:end] for k,v in targets.items()},**loss_options)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite reconstruction loss')
        (loss*weight).backward(retain_graph=end<count)
        total+=float(loss.detach())*weight
        for k,v in values.items():metrics[k]=metrics.get(k,0.)+weight*v
    if torch.distributed.is_initialized():
        # Fixed parameter order, including zero gradients, on every rank.
        world=torch.distributed.get_world_size()
        for parameter in model.parameters():
            if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
            torch.distributed.all_reduce(parameter.grad)
            parameter.grad.div_(world)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();model.eval()
    metrics.update(loss=total,gradient_norm=float(norm))
    return metrics
