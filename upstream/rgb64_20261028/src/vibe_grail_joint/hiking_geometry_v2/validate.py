"""Independent scene test with frozen validation-calibrated thresholds."""
import argparse
import hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--per-group',type=int,default=8);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    args=p.parse_args()
    if args.device.startswith('cuda'):
        from ..gpu_guard import verify_guarded_worker
        verify_guarded_worker()
    import torch
    from ..hiking_perception.contracts import parse_contract,write_json
    from .data import StratifiedSequences
    from .model import SpatialGeometry
    from .metrics import evaluate
    torch.set_num_threads(2)
    state=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    if state.get('architecture')!=SpatialGeometry.architecture:raise ValueError('Not V2 geometry checkpoint')
    geometry,spec=parse_contract(state['contract']);model=SpatialGeometry(geometry,spec).to(args.device)
    model.load_state_dict(state['encoder'])
    training=state['training_contract']
    data=StratifiedSequences(args.data,training['sequence'],args.split)
    if parse_contract(data.contract)[0]!=geometry: raise ValueError('Grid mismatch')
    forbidden=set(state['train_scenes'])|(set(state['val_scenes']) if args.split=='test' else set())
    if set(data.scenes)&forbidden:raise ValueError('Evaluation scene leakage')
    plan=data.fixed_plan(args.per_group,1743)
    report=dict(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                split=args.split,scenes=data.scenes,seed=1743,sequence=training['sequence'],burn_in=training['burn_in'],
                recon_stride=training['recon_stride'],edge_threshold_from_validation=state['calibrated_edge_threshold'],
                dataset_fingerprint=data.fingerprint,plan=data.plan_records(plan),modes={},
                limitation='Stratified offline geometry test; no demonstrated control improvement')
    for mode in ('normal','zero_image','zero_proprio'):
        report['modes'][mode]=evaluate(model,data,plan,args.batch_size,args.device,training['burn_in'],
                                       training['recon_stride'],mode,state['calibrated_edge_threshold'])
    write_json(args.output,report)
    print({m:{k:v for k,v in d.items() if k not in ('groups','edge_thresholds')} for m,d in report['modes'].items()},flush=True)


if __name__=='__main__':main()
