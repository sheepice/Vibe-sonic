"""Persistent collect -> full supervised fit -> independent evaluation pipeline."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from ..hiking_perception.contracts import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--gpu',type=int,default=4)
    args=p.parse_args()
    if args.gpu not in (4,5,6,7):p.error('Physical GPUs 4-7 only')
    root=Path(__file__).resolve().parents[3]
    config_path=Path(args.config).resolve();settings=json.loads(config_path.read_text())
    records=root/settings['record_dir'];records.mkdir(parents=True,exist_ok=False)
    base=[str(root/'.venv/bin/python'),'-m','vibe_grail_joint.hiking_geometry_v2.launch','--gpus',str(args.gpu),'--wait-idle-seconds','43200']
    datasets=[str(root/settings['data_root']/f'seed_{seed}') for seed in settings['seeds']]
    stages=[]
    for seed,directory in zip(settings['seeds'],datasets):
        command=base+['collect','--config',str(root/settings['task_config']),
            '--checkpoint',str(root/settings['collection_policy']),'--output',directory,
            '--seed',str(seed),'--num-envs',str(settings['num_envs']),
            '--steps',str(settings['collection_steps']),'--chunk-steps',str(settings['chunk_steps']),
            '--history-frames',str(settings['history_frames']),'--grid-nx',str(settings['grid_nx']),
            '--grid-ny',str(settings['grid_ny']),'--preview-every','100','--domain-randomization']
        stages.append((f'collect_{seed}',command))
    command=base+['fit','--data',*datasets,'--output',str(root/settings['run_dir']),
        '--warm-start',str(root/settings['warm_start']),'--updates',str(settings['updates']),
        '--batch-size',str(settings['batch_size']),'--sequence',str(settings['sequence']),
        '--burn-in',str(settings['burn_in']),'--recon-stride',str(settings['recon_stride']),
        '--lr',str(settings['learning_rate']),'--eval-every',str(settings['eval_every']),
        '--val-per-group',str(settings['val_per_group'])]
    stages.append(('fit_full',command))
    command=base+['validate','--data',*datasets,'--checkpoint',str(root/settings['run_dir']/'best.pt'),
                 '--output',str(records/'test.json'),'--split','test',
                 '--per-group',str(settings['test_per_group']),'--batch-size',str(settings['batch_size'])]
    stages.append(('test_best',command))
    now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
    status=dict(status='running',supervisor_pid=os.getpid(),physical_gpu=args.gpu,started_at_utc=now(),
                config=str(config_path),settings=settings,stages=[])
    write_json(records/'plan.json',dict(stages=[dict(name=n,command=c) for n,c in stages]))
    try:
        for name,command in stages:
            row=dict(name=name,status='running',started_at_utc=now(),log=str(records/f'{name}.log'))
            status['stages'].append(row);status['current_stage']=name
            write_json(records/'status.json',status)
            with Path(row['log']).open('x') as log:
                proc=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                row['launcher_pid']=proc.pid;write_json(records/'status.json',status)
                code=proc.wait()
            row.update(status='completed' if code==0 else 'failed',exit_code=code,finished_at_utc=now())
            write_json(records/'status.json',status)
            if code:raise RuntimeError(f'{name} failed with exit code {code}; see {row["log"]}')
        status.update(status='completed',finished_at_utc=now())
        write_json(records/'status.json',status)
    except BaseException as error:
        status.update(status='failed',error=repr(error),finished_at_utc=now())
        write_json(records/'status.json',status)
        raise


if __name__=='__main__':main()
