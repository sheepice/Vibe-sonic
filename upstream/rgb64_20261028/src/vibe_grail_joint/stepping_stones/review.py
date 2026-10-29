"""Bounded single-world recording on spare GPU memory; never alters other jobs."""
import argparse,json,os,signal,subprocess,sys,time,fcntl
from pathlib import Path
from ..gpu_guard import query


def verify_review_worker():
    import psutil,torch
    guard=int(os.environ.get('HIKING_REVIEW_GUARD_PID','0'))
    if guard not in [p.pid for p in psutil.Process().parents()]:raise RuntimeError('Use review_shared launcher')
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    if gpu not in (0,1,2,3,6,7):raise RuntimeError('Only physical GPUs 0-3,6,7')
    # Bound torch allocations; the parent independently bounds ALL GPU allocations.
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(3*2**30/total,0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--duration',type=float,default=40.);p.add_argument('--mask-rgb',action='store_true')
    p.add_argument('--seed',type=int,default=271831);p.add_argument('--rgb-delay-steps',type=int,default=0)
    p.add_argument('--terrain-style',choices=('default','same_gray'),default='default')
    p.add_argument('--bundle');p.add_argument('--deploy-mode',choices=('parity','portable'),default='parity')
    p.add_argument('--family',choices=('stairs','box_stones','cylinder_stones','stair_gap'),default='stairs')
    p.add_argument('--level',type=int,default=0)
    p.add_argument('--terrain-variant',type=int,choices=(0,1,2),default=1)
    p.add_argument('--capture-video',action='store_true')
    p.add_argument('--speed',type=float,choices=(.25,.4),default=.25,help='Portable runtime test command speed')
    p.add_argument('--actuator-guards',action='store_true')
    p.add_argument('--stop-hold-seconds',type=float,default=3.)
    p.add_argument('--camera-fps',type=float,default=30.,help='Portable camera frame cadence')
    p.add_argument('--reference-command-mode',choices=('ramped','wanted_step'),
                   help='Isolated diagnostic; changes planner speed only and produces no deployment admission')
    p.add_argument('--record-case',type=int);p.add_argument('--case-report')
    p.add_argument('--probe-case',type=int);p.add_argument('--probe-reference');p.add_argument('--probe-stride',type=int,default=25)
    p.add_argument('--comparison',choices=('baseline','black_rgb'),default='baseline');p.add_argument('--baseline')
    a=p.parse_args()
    if a.gpu not in (0,1,2,3,6,7) or not 0<a.duration<=45:p.error('Only bounded one-world review allowed')
    if a.actuator_guards and (not a.bundle or a.deploy_mode!='portable'):
        p.error('Actuator guards require a portable bundle test')
    if a.capture_video and (not a.bundle or a.deploy_mode!='portable' or not a.actuator_guards):
        p.error('Capture requires a guarded portable test')
    if a.reference_command_mode is not None:
        if (not a.bundle or a.deploy_mode!='portable' or not a.actuator_guards
                or a.capture_video or a.mask_rgb or a.record_case is not None or a.probe_case is not None):
            p.error('Reference diagnostic requires a headless, guarded portable test with original RGB')
    root=Path(__file__).resolve().parents[3]
    config=json.loads(Path(a.config).read_text())
    bank=root/'data/smoke/rank_0/bank.json';meta=json.loads(bank.read_text())
    if meta['total_frames']>100000:raise ValueError('Review requires compact unused replay bank; planner remains online')
    devices={int(r[0]):r for r in query('index,uuid,memory.total,memory.used','gpu')}
    d=devices[a.gpu];free=int(d[2])-int(d[3])
    if free<8192:raise RuntimeError(f'Need at least 8 GiB free, found {free} MiB')
    lock=(root/'records'/f'gpu_{a.gpu}_review.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    audit=Path(a.output+'.resource.json');audit.parent.mkdir(parents=True,exist_ok=True)
    cache=root/'records/hiking_perception_cache'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
             HIKING_REVIEW_GUARD_PID=str(os.getpid()),STEPPING_STONE_SHARED_REVIEW='1',OMP_NUM_THREADS='2',
             HF_HOME=str(root/'models/hf'),HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',
             TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),WARP_CACHE_PATH=str(cache/'warp'),
             MPLCONFIGDIR=str(cache/'matplotlib'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    cmd=[sys.executable,'-m','vibe_grail_joint.stepping_stones.evaluate',
         '--config',str(Path(a.config).resolve()),'--checkpoint',str(Path(a.checkpoint).resolve()),
         '--output',str(Path(a.output).resolve()),'--duration',str(a.duration)]
    cmd+=['--seed',str(a.seed),'--rgb-delay-steps',str(a.rgb_delay_steps),'--terrain-style',a.terrain_style]
    if a.mask_rgb:cmd+=['--mask-rgb']
    if a.bundle:
        if a.mask_rgb:p.error('Deployment contract check does not accept RGB masking')
        cmd=[sys.executable,'-m','vibe_grail_joint.stepping_stones.simcheck',
             '--config',str(Path(a.config).resolve()),'--bundle',str(Path(a.bundle).resolve()),
             '--output',str(Path(a.output).resolve()),'--duration',str(a.duration),
             '--mode',a.deploy_mode,'--family',a.family,'--level',str(a.level),'--speed',str(a.speed),
             '--stop-hold-seconds',str(a.stop_hold_seconds),'--camera-fps',str(a.camera_fps),
             '--seed',str(a.seed),'--terrain-variant',str(a.terrain_variant)]
        if a.actuator_guards:cmd+=['--actuator-guards']
        if a.capture_video:cmd+=['--capture-video']
        if a.reference_command_mode is not None:
            cmd[2]='vibe_grail_joint.stepping_stones.reference_ablation'
            cmd+=['--reference-command-mode',a.reference_command_mode]
    if a.record_case is not None:
        if a.bundle or a.mask_rgb or not a.case_report:p.error('Recording requires --case-report and cannot combine bundle/mask options')
        cmd=[sys.executable,'-m','vibe_grail_joint.stepping_stones.record',
             '--config',str(Path(a.config).resolve()),'--checkpoint',str(Path(a.checkpoint).resolve()),
             '--output',str(Path(a.output).resolve()),'--duration',str(a.duration),
             '--case-report',str(Path(a.case_report).resolve()),'--record-case',str(a.record_case),'--comparison',a.comparison]
        if a.baseline:cmd+=['--baseline',str(Path(a.baseline).resolve())]
    if a.probe_case is not None:
        if a.bundle or a.record_case is not None or a.mask_rgb or not a.case_report or not a.probe_reference:
            p.error('Geometry probe needs --case-report and --probe-reference, without recording/bundle/mask options')
        cmd=[sys.executable,'-m','vibe_grail_joint.stepping_stones.geometry_probe',
             '--config',str(Path(a.config).resolve()),'--checkpoint',str(Path(a.checkpoint).resolve()),
             '--reference-checkpoint',str(Path(a.probe_reference).resolve()),'--case-report',str(Path(a.case_report).resolve()),
             '--case',str(a.probe_case),'--output',str(Path(a.output).resolve()),
             '--duration',str(a.duration),'--stride',str(a.probe_stride)]
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
