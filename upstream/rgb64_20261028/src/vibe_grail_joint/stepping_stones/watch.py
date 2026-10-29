"""Observe known training handles; evaluate immutable checkpoints without restarting jobs."""
from pathlib import Path
import argparse,json,time,subprocess,sys,datetime
import psutil

def main():
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);a=p.parse_args()
    out=Path(a.records).resolve();root=Path(__file__).resolve().parents[3]
    jobs=[];running={};finished=set();failures=[]
    def write():
        status=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training=[],evaluations=list(finished),failures=failures)
        for j in jobs:
            alive=False
            try:
                proc=psutil.Process(j['pid']);alive=proc.is_running() and 'vibe_grail_joint.stepping_stones.launch' in proc.cmdline()
            except psutil.NoSuchProcess:pass
            ranks=[]
            for r in range(2):
                path=Path(j.get('log_dir',root/f'runs/stones_{j["name"]}_20261025'))/f'rank_{r}_status.json'
                if path.exists():
                    d=json.loads(path.read_text());ranks.append({k:d.get(k) for k in ('rank','status','completed_updates','absolute_completed_updates','device_used_gib','updated_at_utc','error')})
            status['training'].append(dict(name=j['name'],launcher_pid=j['pid'],verified_live=alive,ranks=ranks))
        (out/'watch_status.json').write_text(json.dumps(status,indent=2))
        return status
    while True:
        jobs=json.loads((out/'training_jobs.json').read_text())
        status=write()
        for key,(proc,log) in list(running.items()):
            code=proc.poll()
            if code is None:continue
            log.close();del running[key]
            if code==0:finished.add(key)
            else:failures.append(dict(evaluation=key,exit_code=code));finished.add(key)
        for j in jobs:
            name=j['name']
            if any(k.startswith(name+'/') for k in running):continue
            # Save every 250; early checkpoint benchmark then periodic long-run checks.
            for it in j.get('evaluate_iterations',(100,250,500,1000,3000,6000,11999)):
                key=f'{name}/{it}'
                if key in finished:continue
                checkpoint=Path(j.get('log_dir',root/f'runs/stones_{name}_20261025'))/f'model_{it}.pt'
                if not checkpoint.is_file():continue # atomic training saves
                dest=out/f'evaluation_{name}_{it}'
                if dest.exists():finished.add(key);continue
                gpu=int(j.get('evaluation_gpu',j['gpus'].split(',')[0]))
                cmd=[sys.executable,'-m','vibe_grail_joint.stepping_stones.review','--gpu',str(gpu),
                     '--config',str(j.get('config',root/f'configs/stones_{name}_20261025.json')),'--checkpoint',str(checkpoint),'--output',str(dest)]
                log=(out/f'evaluation_{name}_{it}.log').open('w')
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=root,start_new_session=True)
                running[key]=(proc,log);break
        if not any(j['verified_live'] for j in status['training']) and not running:
            write();return
        time.sleep(30)
if __name__=='__main__':main()
