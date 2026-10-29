"""Train high-resolution Hiking geometry with stratified checkpoint selection."""
import argparse
from dataclasses import replace
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--warm-start');p.add_argument('--resume')
    p.add_argument('--updates',type=int,default=15000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--sequence',type=int,default=64);p.add_argument('--burn-in',type=int,default=16)
    p.add_argument('--recon-stride',type=int,default=4);p.add_argument('--history-weight',type=float,default=.25)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=137)
    p.add_argument('--eval-every',type=int,default=250);p.add_argument('--val-per-group',type=int,default=2)
    p.add_argument('--probe',action='store_true');p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    args=p.parse_args()
    if args.device.startswith('cuda'):
        from ..gpu_guard import verify_guarded_worker
        verify_guarded_worker()
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from ..hiking_perception.contracts import parse_contract,contract,write_json
    from .model import SpatialGeometry,terrain_loss
    from .data import StratifiedSequences
    from .metrics import evaluate,calibrate
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    if not 0<=args.burn_in<args.sequence or min(args.updates,args.batch_size,args.recon_stride,args.eval_every,args.val_per_group)<1:
        p.error('Invalid training dimensions')
    if args.history_weight<0 or args.lr<=0: p.error('Invalid loss weight or learning rate')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=bool(args.resume))
    train=StratifiedSequences(args.data,args.sequence,'train');val=StratifiedSequences(args.data,args.sequence,'val')
    if set(train.scenes)&set(val.scenes): raise RuntimeError('Scene leakage')
    geometry,spec=parse_contract(train.contract);spec=replace(spec,temporal=not args.probe)
    model=SpatialGeometry(geometry,spec).to(args.device)
    initialized=[];effective_scenes=set(train.scenes)
    if args.warm_start:
        previous=torch.load(args.warm_start,map_location='cpu',weights_only=False)
        if previous.get('architecture')==SpatialGeometry.architecture:
            if parse_contract(previous['contract'])!=(geometry,spec): raise ValueError('Warm start contract mismatch')
            model.load_state_dict(previous['encoder']);initialized=list(model.state_dict())
        else: initialized=model.warm_start_backbone(previous)
        effective_scenes.update(previous.get('train_scenes',[]))
    if effective_scenes&set(val.scenes): raise RuntimeError('Warm start leaks validation scenes')
    training_contract={k:getattr(args,k) for k in ['sequence','burn_in','recon_stride','history_weight','lr','batch_size','val_per_group','probe']}
    training_contract.update(loss='ce_edgeaware_gradient_v2',sampling='source_route_speed_stage_downweight1p5_v2')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    rng=torch.Generator().manual_seed(args.seed);start=0;best=float('inf')
    plan=val.fixed_plan(args.val_per_group,args.seed+1)
    if args.resume:
        previous=torch.load(args.resume,map_location=args.device,weights_only=False)
        if previous['training_contract']!=training_contract or previous['dataset_fingerprint']!=train.fingerprint:
            raise ValueError('Resume settings/data mismatch')
        if previous['architecture']!=model.architecture or parse_contract(previous['contract'])!=(geometry,spec): raise ValueError('Resume architecture mismatch')
        model.load_state_dict(previous['encoder']);optimizer.load_state_dict(previous['optimizer'])
        rng.set_state(previous['generator'].cpu());start=previous['update']+1;best=previous['best_validation']
        effective_scenes.update(previous['train_scenes'])
    write_json(out/'experiment.json',dict(args=vars(args),architecture=model.architecture,contract=contract(geometry,spec),
        training_contract=training_contract,dataset_fingerprint=train.fingerprint,train_windows=len(train.windows),
        validation_windows=len(plan),train_groups=len(train.groups),val_groups=len(val.groups),
        parameters=sum(v.numel() for v in model.parameters()),warm_started_keys=initialized,
        train_scenes=sorted(effective_scenes),validation_scenes=val.scenes,
        selection='mean stratum height MAE + 0.05*(1-mean supported-stratum edge F1 at threshold 0.5)'))
    write_json(out/'validation_plan.json',val.plan_records(plan))
    writer=SummaryWriter(str(out/'tensorboard'))
    try:
        for update in range(start,start+args.updates):
            model.train();b=train.sample(args.batch_size,rng,args.device)
            h,_=model.sequence(b['tokens'],b['proprio'],b['age'],b['reset'])
            sl=slice(args.burn_in,None,args.recon_stride)
            targets={k:b[k][sl].flatten(0,1) for k in ('height','edge','visible')}
            targets['visible']=targets['visible'] & b['valid'][sl].flatten()[:,None]
            loss,metrics=terrain_loss(model,h[sl].flatten(0,1),targets)
            if args.history_weight and b['historical_visible'][sl].any():
                targets['visible']=b['historical_visible'][sl].flatten(0,1)&b['valid'][sl].flatten()[:,None]
                history,hm=terrain_loss(model,h[sl].flatten(0,1),targets)
                loss=loss+args.history_weight*history;metrics.update(history_height_mae_m=hm['height_mae_m'])
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite geometry loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            metrics.update(total_loss=float(loss.detach()),gradient_norm=float(norm))
            for key,value in metrics.items(): writer.add_scalar('train/'+key,value,update)
            if update%args.eval_every==0 or update==start+args.updates-1:
                vm=evaluate(model,val,plan,args.batch_size,args.device,args.burn_in,args.recon_stride)
                improved=vm['selection_score']<best;best=min(best,vm['selection_score'])
                threshold=calibrate(vm)
                state=dict(schema=2,architecture=model.architecture,contract=contract(geometry,spec),encoder=model.state_dict(),
                    optimizer=optimizer.state_dict(),generator=rng.get_state(),update=update,best_validation=best,
                    calibrated_edge_threshold=threshold,training_contract=training_contract,dataset_fingerprint=train.fingerprint,
                    train_scenes=sorted(effective_scenes),val_scenes=val.scenes,scene_key_version=2,
                    requires_v2_loader=True,control_policy_trained=False)
                names=['last.pt']+(['best.pt'] if improved else [])+([f'model_{update}.pt'] if update%1000==0 else [])
                for name in names:
                    tmp=out/(name+'.tmp');torch.save(state,tmp);tmp.replace(out/name)
                write_json(out/'validation_latest.json',vm)
                if improved: write_json(out/'validation_best.json',vm)
                for key in ['height_mae_m','height_p95_m','macro_height_mae_m','macro_edge_f1_at_05','selection_score']:
                    writer.add_scalar('validation/'+key,vm[key],update)
                report=dict(update=update,train=metrics,validation={k:v for k,v in vm.items() if k not in ('groups','edge_thresholds')},
                    validation_groups=len(vm['groups']),edge_at_05=vm['edge_thresholds']['0.5'],calibrated_threshold=threshold,
                    best_validation=best,cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.device.startswith('cuda') else 0.)
                write_json(out/'status.json',report)
                with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(report)+'\n')
                print(json.dumps(report),flush=True);writer.flush()
    finally:writer.close()


if __name__=='__main__':main()
