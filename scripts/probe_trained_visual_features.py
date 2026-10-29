"""Read-only ONNX taps: frozen encoder versus the three trained spatial blocks.

No original model is edited. Graph slices and additional outputs exist only in
memory. Uses the cached pure protobuf schema without installing ONNX or PyTorch.
"""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont
from rosbags.highlevel import AnyReader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sim2sim.vendor.runtime import Runtime
OUT=ROOT/'records/feature_layers_20261029';OUT.mkdir(exist_ok=True)
schema=next((Path.home()/'.cache/uv/archive-v0').glob('*/onnx/onnx_ml_pb2.py'))
spec=importlib.util.spec_from_file_location('feature_probe_proto',schema)
pb=importlib.util.module_from_spec(spec);spec.loader.exec_module(pb)
policy_path=ROOT/'models/vggt_hierarchical_rgb64/policy.onnx'
binary=policy_path.read_bytes();original=pb.ModelProto();original.ParseFromString(binary)
taps=[f'/extractors.0/blocks.{i}/Add_2_output_0' for i in range(3)]
def output(graph,name,shape):
    v=graph.output.add();v.name=name;v.type.tensor_type.elem_type=1
    for size in shape:v.type.tensor_type.shape.dim.add().dim_value=size
def session(model):
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.log_severity_level=3
    return ort.InferenceSession(model.SerializeToString(),sess_options=opt,providers=['CPUExecutionProvider'])

producer={v:i for i,n in enumerate(original.graph.node) for v in n.output}
seen=set();needed=set()
def visit(name):
    if name in seen:return
    seen.add(name)
    if name in producer:
        i=producer[name];needed.add(i)
        for x in original.graph.node[i].input:visit(x)
for name in taps:visit(name)
spatial=pb.ModelProto();spatial.CopyFrom(original)
for field in ['node','initializer','input','output','value_info']:spatial.graph.ClearField(field)
spatial.graph.node.extend([n for i,n in enumerate(original.graph.node) if i in needed])
spatial.graph.initializer.extend([v for v in original.graph.initializer if v.name in seen])
spatial.graph.input.extend([v for v in original.graph.input if v.name in seen])
for name in taps:output(spatial.graph,name,[1,28,192])
blocks=session(spatial);assert [x.name for x in blocks.get_inputs()]==['kv_tokens__img_tokens']

rt=Runtime(ROOT/'models/vggt_hierarchical_rgb64')
real=ROOT/'records/g1_recordings/trial_20261028_235905/preview_checked_20261029'
sim=ROOT/'records/g1_sim_compare_20261029/sim_15cm_30cm_04/preview'
selected=[];features=[];images=[]
for directory,times,origin,label in [(real,[19.8,22.5,24.8],3151.901268043,'Real'),(sim,[4.,6.4,7.2],0,'Sim')]:
    p=json.loads((directory/'preview.json').read_text())
    candidates=[f for f in p['frames'] if f.get('attention')]
    for t in times:
        f=min(candidates,key=lambda f:abs(f['step']['inference_at']-origin-t))
        rgb=np.array(Image.open(directory/f['policy']).convert('RGB'))
        patches,cls=rt.encode(rgb[:,:,::-1].copy())
        stages=blocks.run(None,{'kv_tokens__img_tokens':patches[None]})
        features.append(np.stack([patches,*[x[0] for x in stages]]));images.append(rgb)
        selected.append(dict(source=label,time=f['step']['inference_at']-origin,policy=str((directory/f['policy']).relative_to(ROOT))))
features=np.stack(features)
# Verify taps using a real recorded SIM policy observation, not dummy states.
bag=ROOT/'records/g1_sim_compare_20261029/sim_15cm_30cm_04/bag'
with AnyReader([bag]) as reader:
    for c,stamp,raw in reader.messages(connections=[c for c in reader.connections if c.topic=='/vibe/policy/step']):
        step=json.loads(reader.deserialize(raw,c.msgtype).data)
        if step['step_id']==320:break
    feed={}
    for c,_,raw in reader.messages(start=stamp,stop=stamp+1):
        if c.topic.startswith('/vibe/policy/inputs/'):
            m=reader.deserialize(raw,c.msgtype);feed[c.topic.split('/')[-1]]=np.array(m.data,np.float32).reshape([d.size for d in m.layout.dim])
cross='/extractors.0/attention/MatMul_1_output_0'
latent='/extractors.0/attention/out_norm/LayerNormalization_output_0'
full=pb.ModelProto();full.CopyFrom(original)
for name in taps:output(full.graph,name,[1,28,192])
output(full.graph,cross,[1,3,64]);output(full.graph,latent,[1,128])
check=session(full)
values=check.run(['actions','attn',*taps,cross,latent],feed)
action_error=float(np.max(abs(values[0][0]-rt.act(feed)[0])))
block_error=max(float(np.max(abs(a-b))) for a,b in zip(values[2:5],blocks.run(None,{'kv_tokens__img_tokens':feed['kv_tokens__img_tokens']})))
assert action_error<1e-5 and block_error<1e-5,(action_error,block_error)

# One PCA basis and fixed color scales across all six images/four stages.
# Normalize each patch by L2 first: PCA represents feature directions, not norms.
unit=features/np.maximum(np.linalg.norm(features,axis=-1,keepdims=True),1e-12)
flat=unit.reshape(-1,192);mean=flat.mean(0)
_,singular,vt=np.linalg.svd(flat-mean,full_matrices=False)
for k in range(3):
    if vt[k,np.argmax(abs(vt[k]))]<0:vt[k]*=-1
projected=(unit-mean)@vt[:3].T
lo,hi=np.percentile(projected.reshape(-1,3),[2,98],axis=0)
colors=np.clip((projected-lo)/np.maximum(hi-lo,1e-12),0,1)
np.savez_compressed(OUT/'features_and_pca.npz',features=features,unit_features=unit,pca_mean=mean,pca_basis=vt[:3],pca_low=lo,pca_high=hi)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
canvas=Image.new('RGB',(1120,180*len(selected)),(20,27,34));draw=ImageDraw.Draw(canvas)
for row,(record,rgb,color) in enumerate(zip(selected,images,colors)):
    y=row*180
    for col,title in enumerate(['Policy input','Frozen encoder: PCA','Trained block 1: PCA','Trained block 2: PCA','Trained block 3: PCA']):
        draw.text((col*224+5,y+3),title,font=font,fill='white')
        im=Image.fromarray(rgb).resize((224,128),Image.Resampling.NEAREST) if col==0 else Image.fromarray(np.round(color[col-1].reshape(4,7,3)*255).astype(np.uint8)).resize((224,128),Image.Resampling.NEAREST)
        canvas.paste(im,(col*224,y+25))
    draw.text((5,y+158),f"{record['source']} t={record['time']:.2f}s | Shared PCA basis and scale; colors are NOT attention or semantic labels.",font=font,fill='#bbccd8')
canvas.save(OUT/'trained_feature_comparison.png')
summary=dict(policy_sha256=hashlib.sha256(binary).hexdigest(),original_model_unchanged=policy_path.read_bytes()==binary,
    selected=selected,spatial_outputs=dict(zip(taps,[list(x.shape) for x in values[2:5]])),
    attention_shape=list(values[1].shape),cross_attention_output_shape=list(values[-2].shape),
    after_layernorm_shape=list(values[-1].shape),action_shape=list(values[0].shape),
    action_error_with_extra_outputs=action_error,sliced_spatial_graph_error=block_error,
    pca_explained_variance_ratio=(singular[:3]**2/(singular**2).sum()).tolist(),
    feature_norm_median_by_stage=np.median(np.linalg.norm(features,axis=-1),axis=(0,2)).tolist(),
    note='Frozen encoder is not fine-tuned. Trained block outputs retain 28 spatial slots; cross-attention summaries and 128-D latent do not. PCA colors show feature similarity, not relevance or edge recognition.')
(OUT/'layer_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='selected'},ensure_ascii=False,indent=2))

# Attribute changes at the user's proposed late layers back to image regions,
# rather than reshaping the non-spatial 128-D/29-D outputs into a fake image.
probes=[];probe_images=[]
with AnyReader([bag]) as reader:
    requested={280,320,360};selected_steps={}
    for c,stamp,raw in reader.messages(connections=[c for c in reader.connections if c.topic=='/vibe/policy/step']):
        step=json.loads(reader.deserialize(raw,c.msgtype).data)
        if step['step_id'] in requested:selected_steps[step['step_id']]=stamp
    for step_id,stamp in sorted(selected_steps.items()):
        feed={};bgr=None
        for c,_,raw in reader.messages(start=stamp,stop=stamp+1):
            if c.topic.startswith('/vibe/policy/inputs/'):
                m=reader.deserialize(raw,c.msgtype);feed[c.topic.split('/')[-1]]=np.array(m.data,np.float32).reshape([d.size for d in m.layout.dim])
            elif c.topic=='/vibe/camera/policy/image':
                m=reader.deserialize(raw,c.msgtype);bgr=np.array(m.data).reshape(64,112,3)
        action0,z0,attn=check.run(['actions',latent,'attn'],feed)
        dz=[];da=[]
        for patch in range(28):
            perturbed=bgr.copy();y,x=divmod(patch,7)
            perturbed[16*y:16*(y+1),16*x:16*(x+1)]=bgr.mean(axis=(0,1)).astype(np.uint8)
            tokens,cls=rt.encode(perturbed)
            a,z=check.run(['actions',latent],dict(feed,kv_tokens__img_tokens=tokens[None],q_cls=cls[None]))
            dz.append(float(np.sqrt(np.mean((z-z0)**2))))
            da.append(float(np.sqrt(np.mean(((a-action0)[0,:12]*rt.scale[:12])**2))))
        probes.append(dict(step_id=step_id,time_s=step_id*.02,mean_attention=attn[0].mean(0).tolist(),
            after_layernorm_change_rmse=dz,final_leg_target_change_rmse_rad=da))
        probe_images.append(bgr[:,:,::-1].copy())
old=json.loads((ROOT/'records/g1_sim_compare_20261029/sim_occlusion.json').read_text())
old_error=max(float(np.max(abs(np.array(p['final_leg_target_change_rmse_rad'])-o['scores_leg_target_rmse_rad']))) for p,o in zip(probes,old['frames']))
assert old_error<1e-5,old_error
stops=np.array([[22,26,90],[22,95,220],[22,211,192],[255,231,60],[240,45,35]])
def heat_overlay(rgb,values):
    values=np.asarray(values,np.float32).reshape(4,7);values/=max(float(values.max()),1e-12)
    display=np.asarray(Image.fromarray(values).resize((280,160),Image.Resampling.BILINEAR))
    heat=np.stack([np.interp(display,np.linspace(0,1,5),stops[:,j]) for j in range(3)],axis=-1)
    base=np.asarray(Image.fromarray(rgb).resize((280,160),Image.Resampling.BILINEAR))
    return Image.fromarray(np.round(base*.45+heat*.55).astype(np.uint8))
canvas=Image.new('RGB',(1120,230*len(probes)),(20,27,34));draw=ImageDraw.Draw(canvas)
for row,(p,rgb) in enumerate(zip(probes,probe_images)):
    y=row*230;draw.text((5,y+3),f"SIM t={p['time_s']:.1f}s | Fixed state/history/reference; mask RGB patch, recompute patch + CLS",font=font,fill='white')
    columns=[('Policy image',None),('Current mean attention',p['mean_attention']),
        ('Post-LayerNorm sensitivity',p['after_layernorm_change_rmse']),('Final leg-target sensitivity',p['final_leg_target_change_rmse_rad'])]
    for col,(title,data) in enumerate(columns):
        draw.text((col*280+5,y+25),title,font=font,fill='white')
        im=Image.fromarray(rgb).resize((280,160),Image.Resampling.NEAREST) if data is None else heat_overlay(rgb,data)
        canvas.paste(im,(col*280,y+47))
        if data is not None:draw.text((col*280+5,y+209),f"display 0..{max(data):.5f}"+(' rad' if col==3 else ''),font=font,fill='#bbccd8')
canvas.save(OUT/'late_output_sensitivity.png')
(OUT/'late_output_sensitivity.json').write_text(json.dumps(dict(probes=probes,old_action_probe_max_error=old_error,
    method='One 16x16 RGB patch replaced by frame mean, re-encode patch and CLS, all nonvisual inputs fixed; compute post-LayerNorm latent RMSE and final scaled leg-target RMSE.',
    caveat='Frozen-state sensitivity only; not a semantic edge detector or a closed-loop success attribution. Color scale is relative per panel; real full policy inputs unavailable locally.'),indent=2)+'\n')
print('Saved trained-feature comparison and downstream-output sensitivity; action probe repeat error:',old_error)
