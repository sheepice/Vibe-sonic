"""Held-out scene geometry and action-conditional prediction validation."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True); p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--batches',type=int,default=40); p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    args=p.parse_args()
    if args.device.startswith('cuda'):
        from ..gpu_guard import verify_guarded_worker
        verify_guarded_worker()
    import torch
    from .dataset import SequenceDataset
    from .contracts import parse_contract,write_json
    from .models import GeometryMemory,ActionWorldModel,future_mask
    from .geometry import decode_height,height_bins
    torch.set_num_threads(2)
    state=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    g,m=parse_contract(state['contract']); encoder=GeometryMemory(g,m).to(args.device).eval()
    encoder.load_state_dict(state['encoder'])
    world=ActionWorldModel(m).to(args.device).eval() if state.get('world') is not None else None
    if world is not None: world.load_state_dict(state['world'])
    dataset=SequenceDataset(args.data,32,args.split)
    if parse_contract(dataset.contract)[0] != g: raise ValueError('Geometry grid mismatch')
    report=dict(checkpoint=str(Path(args.checkpoint).resolve()),split=args.split,scenes=dataset.scenes,
                windows=len(dataset.windows),sampled_windows=args.batches*args.batch_size,seed=1743,
                modes={},limitation='Finite held-out sample; supervised metrics do not establish safer locomotion')
    modes=('normal','zero_image','zero_proprio','shuffled_actions') if world else ('normal','zero_image','zero_proprio')
    with torch.no_grad():
        for mode in modes:
            rng=torch.Generator().manual_seed(1743); errors=[]; historical_errors=[]; counts=torch.zeros(4,dtype=torch.long)
            state_sse=torch.zeros(3); state_count=torch.zeros(3); risk_counts=torch.zeros(3,4,dtype=torch.long)
            onset_counts=torch.zeros_like(risk_counts)
            for _ in range(args.batches):
                b=dataset.sample(args.batch_size,rng,args.device)
                if mode=='zero_image': b['tokens'].zero_()
                if mode=='zero_proprio': b['proprio'].zero_()
                h,_=encoder.sequence(b['tokens'],b['proprio'],b['age'],b['reset'])
                if world is None:
                    logits,offset,edge=encoder.heads(h[8:].flatten(0,1)); height=b['height'][8:].flatten(0,1)
                    mask=b['visible'][8:].flatten(0,1)&height_bins(height,g)[2]
                    error=(decode_height(logits,offset,g)-height).abs(); errors.append(error[mask].cpu())
                    if 'historical_visible' in b:
                        historical_errors.append(error[b['historical_visible'][8:].flatten(0,1)&height_bins(height,g)[2]].cpu())
                    pred=edge.sigmoid()>.5; true=b['edge'][8:].flatten(0,1).bool()
                    counts+=torch.tensor([int((pred&true&mask).sum()),int((pred&~true&mask).sum()),int((~pred&true&mask).sum()),int((~pred&~true&mask).sum())])
                else:
                    start=8; stop=start+max(m.horizon_steps); actions=b['action'][start:stop]
                    if mode=='shuffled_actions': actions=actions.flip(1)
                    if state.get('action_mode')=='zero': actions=torch.zeros_like(actions)
                    pred,risk=world(h[start],actions,b['command'][start:stop],b['reference_twist'][start:stop])
                    valid=future_mask(b['done'][start:stop],m.horizon_steps)
                    for j,k in enumerate(m.horizon_steps):
                        err=(pred[k-1]-b['next_state'][start+k-1]).square().mean(-1)
                        state_sse[j]+=err[valid[j]].sum().cpu(); state_count[j]+=valid[j].sum().cpu()
                        pr=risk[:k].amax(0).sigmoid()>.5; tr=b['contact'][start:start+k].amax(0)>.5; mask=valid[j,:,None]
                        risk_counts[j]+=torch.tensor([int((pr&tr&mask).sum()),int((pr&~tr&mask).sum()),int((~pr&tr&mask).sum()),int((~pr&~tr&mask).sum())])
                        onset_mask=mask & ~b['contact'][:start].bool().any(0)
                        onset_counts[j]+=torch.tensor([int((pr&tr&onset_mask).sum()),int((pr&~tr&onset_mask).sum()),int((~pr&tr&onset_mask).sum()),int((~pr&~tr&onset_mask).sum())])
            if world is None:
                e=torch.cat(errors)
                report['modes'][mode]=dict(height_mae_m=float(e.mean()),height_p95_m=float(e.quantile(.95)),valid_cells=len(e),
                    edge_confusion_tp_fp_fn_tn=counts.tolist(),edge_recall=float(counts[0]/(counts[0]+counts[2]).clamp_min(1)),
                    edge_precision=float(counts[0]/(counts[0]+counts[1]).clamp_min(1)))
                he=torch.cat(historical_errors) if historical_errors else torch.empty(0)
                report['modes'][mode].update(historical_cells=len(he),historical_height_mae_m=float(he.mean()) if len(he) else None)
            else:
                report['modes'][mode]=[dict(horizon_s=k*m.control_dt,state_mse=float(state_sse[j]/state_count[j].clamp_min(1)),
                    confusion_tp_fp_fn_tn=risk_counts[j].tolist(),collision_positives=int(risk_counts[j,0]+risk_counts[j,2]),
                    recall=float(risk_counts[j,0]/(risk_counts[j,0]+risk_counts[j,2]).clamp_min(1)),
                    false_positive_rate=float(risk_counts[j,1]/(risk_counts[j,1]+risk_counts[j,3]).clamp_min(1)),
                    no_contact_in_previous_s=.16,onset_confusion_tp_fp_fn_tn=onset_counts[j].tolist(),
                    onset_recall=float(onset_counts[j,0]/(onset_counts[j,0]+onset_counts[j,2]).clamp_min(1))) for j,k in enumerate(m.horizon_steps)]
    write_json(args.output,report); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
