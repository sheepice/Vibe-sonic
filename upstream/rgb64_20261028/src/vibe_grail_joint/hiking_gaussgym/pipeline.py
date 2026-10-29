"""Persistent RGB collection -> GaussGym pretrain -> online ViBe PPO -> evaluation."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
from ..hiking_perception.contracts import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--gpu',type=int,default=4)
    a=p.parse_args()
    if a.gpu not in (4,5,6,7):p.error('Only physical GPUs 4-7')
    root=Path(__file__).resolve().parents[3];config=Path(a.config).resolve();c=json.loads(config.read_text())
    records=root/c['record_dir'];records.mkdir(parents=True,exist_ok=False)
    base=[sys.executable,'-m','vibe_grail_joint.hiking_gaussgym.launch','--gpus',str(a.gpu),'--wait-idle-seconds','43200']
    datasets=[str(root/c['data_root']/f'seed_{seed}') for seed in c['seeds']]
    stages=[]
    for seed,directory in zip(c['seeds'],datasets):
        stages.append((f'collect_{seed}',base+['collect','--config',str(root/c['collection_config']),
          '--checkpoint',str(root/c['collection_policy']),'--output',directory,'--seed',str(seed),
          '--num-envs',str(c['collection_envs']),'--steps',str(c['collection_steps'])]))
    stages.append(('geometry_pretrain',base+['fit','--data',*datasets,'--output',str(root/c['geometry_run']),
          '--updates',str(c['geometry_updates']),'--batch-size',str(c['geometry_batch_size']),
          '--eval-every',str(c['eval_every']),'--microbatch',str(c['microbatch'])]))
    stages.append(('geometry_heldout',base+['validate','--data',*datasets,
          '--checkpoint',str(root/c['geometry_run']/'best.pt'),'--output',str(records/'geometry_test.json')]))
    stages.append(('online_control_train',base+['train','--config',str(root/c['control_config']),
          '--log-dir',str(root/c['control_run']),'--num-envs',str(c['control_envs']),
          '--iterations',str(c['control_iterations']),'--mode','geometry',
          '--perception-checkpoint',str(root/c['geometry_run']/'best.pt')]))
    policy=str(root/c['control_run']/f"model_{c['control_iterations']-1}.pt")
    for intervention in ('normal','zero_tokens','zero_geometry'):
        stages.append((f'control_eval_{intervention}',base+['evaluate','--config',str(root/c['control_config']),
          '--checkpoint',policy,'--output',str(records/f'eval_{intervention}'),'--duration','40',
          '--intervention',intervention]))
    now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
    status=dict(status='running',supervisor_pid=os.getpid(),physical_gpu=a.gpu,config=str(config),settings=c,started_at_utc=now(),stages=[])
    write_json(records/'plan.json',dict(stages=[dict(name=n,command=cmd) for n,cmd in stages]))
    try:
        for name,command in stages:
            row=dict(name=name,status='running',started_at_utc=now(),log=str(records/(name+'.log')))
            status['stages'].append(row);status['current_stage']=name;write_json(records/'status.json',status)
            with Path(row['log']).open('x') as log:
                proc=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                row['launcher_pid']=proc.pid;write_json(records/'status.json',status)
                while proc.poll() is None:
                    status['heartbeat_utc']=now();write_json(records/'status.json',status);time.sleep(10)
                code=proc.returncode
            row.update(status='completed' if code==0 else 'failed',exit_code=code,finished_at_utc=now())
            write_json(records/'status.json',status)
            if code:raise RuntimeError(f'{name} exited {code}; see {row["log"]}')
        status.update(status='completed',finished_at_utc=now());write_json(records/'status.json',status)
    except BaseException as e:
        status.update(status='failed',error=repr(e),finished_at_utc=now());write_json(records/'status.json',status);raise

if __name__=='__main__':main()
