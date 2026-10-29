"""Held-out RGB geometry evaluation and RGB ablation, without policy success claims."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',nargs='+',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--per-group',type=int,default=8)
    a=p.parse_args()
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    import torch
    from .data import Sequences
    from .model import load_model
    from .metrics import evaluate
    from ..hiking_perception.contracts import write_json
    torch.set_num_threads(2)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);settings=state['args']
    data=Sequences(a.data,settings['sequence'],a.split)
    if set(data.scenes)&set(state['train_scenes']):raise RuntimeError('Evaluation overlaps training scenes')
    model=load_model(state,'cuda:0').eval();plan=data.fixed_plan(a.per_group,811)
    result=dict(checkpoint=str(Path(a.checkpoint).resolve()),update=state['update'],scenes=data.scenes,plan=data.plan_records(plan))
    for mode in ('normal','zero_rgb','zero_proprio'):
        result[mode]=evaluate(model,data,plan,'cuda:0',settings['burn_in'],settings['recon_stride'],mode)
        write_json(a.output,result)
    print({mode:result[mode]['height_mae_m'] for mode in ('normal','zero_rgb','zero_proprio')},flush=True)

if __name__=='__main__':main()
