"""Bounded single-world recording on spare GPU memory; never alters other jobs."""
import argparse,json,os,signal,subprocess,sys,time,fcntl
from pathlib import Path
from ..gpu_guard import query


def verify_review_worker():
    import psutil,torch
    guard=int(os.environ.get('HIKING_REVIEW_GUARD_PID','0'))
    if guard not in [p.pid for p in psutil.Process().parents()]:raise RuntimeError('Use review_shared launcher')
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    if gpu not in (4,5,6,7):raise RuntimeError('Only physical GPUs 4-7')
    # Bound torch allocations; the parent independently bounds ALL GPU allocations.
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(3*2**30/total,0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--record-case',type=int,default=4);p.add_argument('--duration',type=float,default=40.)
    a=p.parse_args()
    if a.gpu not in (4,5,6,7) or not 0<=a.record_case<48 or not 0<a.duration<=45:p.error('Only bounded one-world review allowed')
    root=Path(__file__).resolve().parents[3]
    config=json.loads(Path(a.config).read_text())
    bank=Path(config['bank_root'])/'rank_0/bank.json';meta=json.loads(bank.read_text())
    if meta['total_frames']>100000:raise ValueError('Review requires compact unused replay bank; planner remains online')
    devices={int(r[0]):r for r in query('index,uuid,memory.total,memory.used','gpu')}
    d=devices[a.gpu];free=int(d[2])-int(d[3])
    if free<8192:raise RuntimeError(f'Need at least 8 GiB free, found {free} MiB')
    lock=(root/'records'/f'gpu_{a.gpu}_review.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    audit=Path(a.output+'.resource.json');audit.parent.mkdir(parents=True,exist_ok=True)
    cache=root/'records/hiking_perception_cache'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
             HIKING_REVIEW_GUARD_PID=str(os.getpid()),OMP_NUM_THREADS='2',
             HF_HOME=str(root/'models/hf'),HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',
             TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),WARP_CACHE_PATH=str(cache/'warp'),
             MPLCONFIGDIR=str(cache/'matplotlib'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    cmd=[sys.executable,'-m','vibe_grail_joint.hiking_gaussgym.long_route_evaluate',
         '--config',str(Path(a.config).resolve()),'--checkpoint',str(Path(a.checkpoint).resolve()),
         '--output',str(Path(a.output).resolve()),'--record-case',str(a.record_case),'--duration',str(a.duration),'--num-envs','2']
    before=[r for r in query('gpu_uuid,pid,used_gpu_memory','compute-apps') if r[0]==d[1]]
    record=dict(physical_gpu=a.gpu,initial_free_mib=free,existing_processes=before,max_review_mib=0,
                all_allocator_limit_mib=4096,torch_allocator_limit_gib=3,command=cmd,status='running')
    proc=subprocess.Popen(cmd,env=env,cwd=root,start_new_session=True);record['worker_pid']=proc.pid
    start=time.monotonic()
    try:
        while proc.poll() is None:
            own=[r for r in query('gpu_uuid,pid,used_gpu_memory','compute-apps') if r[0]==d[1] and int(r[1])==proc.pid]
            used=sum(int(r[2]) for r in own);record['max_review_mib']=max(record['max_review_mib'],used)
            current={int(r[0]):r for r in query('index,memory.total,memory.used','gpu')}[a.gpu]
            free_now=int(current[1])-int(current[2])
            if used>4096 or free_now<4096 or time.monotonic()-start>1800:
                record['status']='stopped_own_review_budget';os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise RuntimeError('Own review stopped to preserve resource margin')
            audit.write_text(json.dumps(record,indent=2)+'\n');time.sleep(3)
        record['exit_code']=proc.returncode;record['status']='completed' if proc.returncode==0 else 'failed'
        audit.write_text(json.dumps(record,indent=2)+'\n')
        raise SystemExit(proc.returncode)
    finally:
        audit.write_text(json.dumps(record,indent=2)+'\n');lock.close()

if __name__=='__main__':main()
