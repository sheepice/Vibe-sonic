"""Frozen-state patch occlusion for SIMULATION inputs; no causal real-robot claim."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from rosbags.highlevel import AnyReader
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sim2sim.vendor.runtime import Runtime
OUT=ROOT/'records/g1_sim_compare_20261029'
rt=Runtime(ROOT/'models/vggt_hierarchical_rgb64')
steps={int(t*50):{} for t in [5.6,6.4,7.2]}
def array(m):return np.asarray(m.data,dtype=np.float32).reshape([d.size for d in m.layout.dim])
with AnyReader([OUT/'sim_15cm_30cm_04/bag']) as reader:
    bystamp={}
    for c,stamp,raw in reader.messages(connections=[c for c in reader.connections if c.topic=='/vibe/policy/step']):
        s=json.loads(reader.deserialize(raw,c.msgtype).data)
        if s['step_id'] in steps:bystamp[stamp]=s['step_id']
    for stamp,step in bystamp.items():
        feed={};image=None;recorded_action=None;attention=None
        for c,_,raw in reader.messages(start=stamp,stop=stamp+1):
            m=reader.deserialize(raw,c.msgtype)
            if c.topic.startswith('/vibe/policy/inputs/'):feed[c.topic.split('/')[-1]]=array(m)
            if c.topic=='/vibe/camera/policy/image':image=np.array(m.data).reshape(64,112,3)
            if c.topic=='/vibe/policy/actions':recorded_action=array(m)
            if c.topic=='/vibe/policy/attention':attention=array(m).reshape(3,28)
        a0=rt.act(feed)[0];scores=[]
        patches,cls=rt.encode(image)
        control=dict(feed,kv_tokens__img_tokens=patches[None],q_cls=cls[None])
        control_error=float(np.max(abs(rt.act(control)[0]-a0)))
        for j in range(28):
            changed=image.copy();y,x=divmod(j,7)
            changed[y*16:(y+1)*16,x*16:(x+1)*16]=image.mean(axis=(0,1)).astype(np.uint8)
            patches,cls=rt.encode(changed)
            a=rt.act(dict(feed,kv_tokens__img_tokens=patches[None],q_cls=cls[None]))[0]
            diff=(a-a0)*rt.scale
            scores.append(float(np.sqrt(np.mean(diff[:12]**2))))
        steps[step]=dict(time_s=step/50,scores_leg_target_rmse_rad=scores,
            max_leg_target_rmse_rad=max(scores),recorded_action_replay_error=float(np.max(abs(a0-recorded_action))),
            unmasked_reencode_action_error=control_error,attention=attention.tolist(),
            mean_attention_vs_occlusion_spearman=float(spearmanr(attention.mean(0),scores).statistic))
        Image.fromarray(image[:,:,::-1]).save(OUT/f'occlusion_input_{step}.png')
result=dict(method='Replace one 16x16 BGR patch with image channel means; re-encode BOTH patches and CLS; hold all state/history/reference inputs fixed. Leg target RMSE over first 12 joints. Simulation only.',
    limitations='This measures frozen-state output sensitivity, not closed-loop success or real-robot causation. Input perturbations can be out of distribution.',frames=list(steps.values()))
(OUT/'sim_occlusion.json').write_text(json.dumps(result,indent=2)+'\n')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
canvas=Image.new('RGB',(1008,240*len(steps)),(20,27,34));draw=ImageDraw.Draw(canvas)
for row,(step,res) in enumerate(steps.items()):
    y=row*240;im=Image.open(OUT/f'occlusion_input_{step}.png').resize((336,192),Image.Resampling.NEAREST)
    canvas.paste(im,(0,y+44))
    draw.text((6,y+4),f"Sim t={res['time_s']:.1f}s | frozen-state occlusion | max leg target RMSE={res['max_leg_target_rmse_rad']:.3f} rad",font=font,fill='white')
    for col,title in enumerate(['Policy input','Mean attention (peak normalized)','Patch occlusion (peak normalized)']):draw.text((col*336+6,y+24),title,font=font,fill='white')
    for col,values in [(1,np.array(res['attention']).mean(0)),(2,np.array(res['scores_leg_target_rmse_rad']))]:
        values=values/max(values.max(),1e-20)
        for j,v in enumerate(values):
            xx=col*336+j%7*48;yy=y+44+j//7*48
            draw.rectangle((xx,yy,xx+47,yy+47),fill=(int(30+225*v),int(60+110*v),int(85-65*v)),outline='#7e8f9b')
canvas.save(OUT/'sim_occlusion.jpg',quality=93)
print(json.dumps({k: {n:v for n,v in r.items() if n not in ['attention','scores_leg_target_rmse_rad']} for k,r in steps.items()},indent=2))
