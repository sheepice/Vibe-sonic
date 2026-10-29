"""Verify recording did not change simulation, and re-encode real saved inputs."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sim2sim import engine,scene

OUT=ROOT/'records/g1_sim_compare_20261029'
sim=engine.Simulation(ROOT/'models/vggt_hierarchical_rgb64')
sim.reset('up',.15,3,.4,42,tread=.30,start=1.5)
saved=np.load(OUT/'sim_15cm_30cm_04/telemetry.npz')
errors=[];reached=None
for i in range(len(saved['time'])):
    errors.append(float(np.max(abs(sim.data.qpos[sim.qids].astype(np.float32)-saved['q'][i]))))
    m=sim.step(.4 if i>=100 and reached is None else 0.)
    if m['reached'] and reached is None:reached=m['t']
sim.close_renderers()
preview_dir=ROOT/'records/g1_recordings/trial_20261028_235905/preview_checked_20261029'
p=json.loads((preview_dir/'preview.json').read_text())
diff=[];checks=[]
for f in p['frames']:
    if f.get('attention') is None or f.get('feature_norm') is None:continue
    bgr=np.array(Image.open(preview_dir/f['policy']))[:,:,::-1].copy()
    features,_=sim.runtime.encode(bgr)
    e=float(np.max(abs(np.linalg.norm(features,axis=-1)-f['feature_norm'])))
    diff.append(e);checks.append(dict(step=f['step']['step_id'],max_norm_error=e))
result=dict(uninstrumented_baseline=dict(frames=len(errors),max_q_error_rad=max(errors),
    reached_at_s=reached,fall=sim.fall,bit_identical_q_float32=max(errors)==0),
    real_encoder_check=dict(samples=len(diff),max_feature_norm_error=max(diff),
       median_feature_norm_error=float(np.median(diff)),
       scope='CPU ONNX encoding of saved exact PNG versus recorded TRT feature norms; not full feature-vector or action parity'),
    hashes_match=sim.runtime.manifest['files']==p['metadata']['manifest']['files'],per_frame=checks)
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='per_frame'},indent=2))
