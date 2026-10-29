"""Long RGB geometry pretraining with GaussGym occupancy/centroid/velocity targets."""
import argparse,json,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--updates',type=int,default=30000);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--sequence',type=int,default=64);p.add_argument('--burn-in',type=int,default=16)
    p.add_argument('--recon-stride',type=int,default=8);p.add_argument('--microbatch',type=int,default=2)
    p.add_argument('--eval-every',type=int,default=500);p.add_argument('--val-per-group',type=int,default=2)
    p.add_argument('--resume');p.add_argument('--seed',type=int,default=137)
    a=p.parse_args()
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from ..hiking_perception.contracts import GeometrySpec,write_json
    from .model import GaussGeometry,ARCHITECTURE
    from .data import Sequences
    from .learning import supervised_update
    from .metrics import evaluate
    torch.set_num_threads(2);torch.manual_seed(a.seed);device='cuda:0'
    if not 0<=a.burn_in<a.sequence or min(a.updates,a.batch_size,a.recon_stride,a.microbatch,a.eval_every)<1:p.error('Invalid dimensions')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=bool(a.resume));started=time.monotonic()
    train=Sequences(a.data,a.sequence,'train');val=Sequences(a.data,a.sequence,'val')
    if set(train.scenes)&set(val.scenes):raise RuntimeError('Scene leakage')
    model=GaussGeometry(GeometrySpec(**train.contract['geometry'])).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    rng=torch.Generator().manual_seed(a.seed);plan=val.fixed_plan(a.val_per_group,a.seed+1)
    begin=0;best=float('inf')
    if a.resume:
        state=torch.load(a.resume,map_location=device,weights_only=False)
        if state['contract']!=model.contract() or state['dataset_fingerprint']!=train.fingerprint:raise ValueError('Resume data/model mismatch')
        model.load_state_dict(state['encoder']);optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['rng'].cpu());begin=state['update']+1;best=state['best']
    else:
        # Estimate once and freeze; online PPO uses exactly this normalization.
        for _ in range(128):model.update_normalizer(train.sample(a.batch_size,rng,device)['proprio'])
    experiment=dict(args=vars(a),contract=model.contract(),parameters=sum(p.numel() for p in model.parameters()),
                    train_scenes=train.scenes,val_scenes=val.scenes,dataset_fingerprint=train.fingerprint,
                    input='RGB + original proprio only; no depth/terrain inputs',
                    selection='macro full-grid height MAE across source/route/speed/stage',
                    original_loss='unweighted occupancy BCE + occupied-centroid MSE + body linear/angular velocity MSE')
    write_json(out/'experiment.json',experiment);write_json(out/'validation_plan.json',val.plan_records(plan))
    writer=SummaryWriter(str(out/'tensorboard'))
    try:
        for update in range(begin,begin+a.updates):
            lr=1e-4 if update<100 else 5e-5
            for group in optimizer.param_groups:group['lr']=lr
            metrics=supervised_update(model,train.sample(a.batch_size,rng,device),optimizer,a.burn_in,a.recon_stride,a.microbatch)
            for k,v in metrics.items():writer.add_scalar('train/'+k,v,update)
            report=dict(status='running',update=update,requested_updates=begin+a.updates,train=metrics,
                        elapsed_s=time.monotonic()-started,cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            if update%10==0:write_json(out/'status.json',report);print(json.dumps(report),flush=True)
            if update%a.eval_every==0 or update==begin+a.updates-1:
                vm=evaluate(model,val,plan,device,a.burn_in,a.recon_stride);improved=vm['macro_height_mae_m']<best
                best=min(best,vm['macro_height_mae_m'])
                state=dict(architecture=ARCHITECTURE,contract=model.contract(),encoder=model.state_dict(),optimizer=optimizer.state_dict(),
                           update=update,best=best,rng=rng.get_state(),args=vars(a),train_scenes=train.scenes,
                           val_scenes=val.scenes,dataset_fingerprint=train.fingerprint,control_policy_trained=False)
                for name in ['last.pt']+(['best.pt'] if improved else [])+([f'model_{update}.pt'] if update%2000==0 else []):
                    tmp=out/(name+'.tmp');torch.save(state,tmp);tmp.replace(out/name)
                write_json(out/'validation_latest.json',vm)
                if improved:write_json(out/'validation_best.json',vm)
                report['validation']={k:v for k,v in vm.items() if k!='groups'};report['best']=best
                write_json(out/'status.json',report)
                for k in ('height_mae_m','macro_height_mae_m','height_p95_m','linvel_rmse_m_s','angvel_rmse_rad_s'):writer.add_scalar('validation/'+k,vm[k],update)
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
                print(json.dumps(report),flush=True);writer.flush()
        report['status']='completed';write_json(out/'status.json',report)
    finally:writer.close()

if __name__=='__main__':main()
