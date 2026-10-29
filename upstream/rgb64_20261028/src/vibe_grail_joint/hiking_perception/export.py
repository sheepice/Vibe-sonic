"""Export the RGB Hiking policy plus an explicit-state recurrent ONNX module."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np


def make_graph(actor):
    from rsl_rl.models.extractor_model import _OnnxExtractorSonicAdapterModel
    from rsl_rl.models._onnx_export import Port
    class Graph(_OnnxExtractorSonicAdapterModel):
        def _adapter_ports(self, model):
            ports=super()._adapter_ports(model)
            self.command_encoder=copy.deepcopy(model.command_encoder)
            self.geometry_encoder=copy.deepcopy(model.geometry_encoder)
            dimension=model.geometry_encoder[0].in_features
            return ports+[Port('velocity_command',(8,),('velocity_command',)),Port('hiking_latent',(dimension,),('hiking_latent',))]
        def _adapter_latent(self, inputs):
            return super()._adapter_latent(inputs[:-2])+self.command_encoder(inputs[-2])+self.geometry_encoder(inputs[-1])
    return Graph(actor,with_attn=True).cpu().eval()


def export_memory(encoder,path):
    import torch
    from torch import nn
    from ..deploy.runtime import session
    class MemoryGraph(nn.Module):
        def __init__(self, model):
            super().__init__(); self.model=model
        def forward(self,tokens,proprio,elapsed,hidden,cell,reset):
            return self.model.step(tokens,proprio,elapsed,hidden,cell,reset)
    graph=MemoryGraph(encoder.cpu().eval())
    m=encoder.spec
    h,c=encoder.initial(1)
    dummy=(torch.zeros(1,m.tokens,m.token_dim),torch.zeros(1,m.proprio_dim),torch.full((1,),m.control_dt),h,c,torch.ones(1,dtype=torch.bool))
    names=['tokens','proprio','elapsed','hidden','cell','reset']
    torch.onnx.export(graph,dummy,str(path),opset_version=18,dynamo=False,input_names=names,output_names=['next_hidden','next_cell'])
    sess=session(path,'cpu'); rng=torch.Generator().manual_seed(491)
    maximum=0.; oh,oc=np.zeros((1,m.hidden_dim),np.float32),np.zeros((1,m.hidden_dim),np.float32)
    with torch.no_grad():
        for t in range(64):
            tokens=torch.randn(1,m.tokens,m.token_dim,generator=rng); prop=torch.randn(1,m.proprio_dim,generator=rng)
            elapsed=torch.full((1,),m.control_dt); reset=torch.tensor([t in (0,17,45)])
            feed=dict(zip(names,[tokens.numpy(),prop.numpy(),elapsed.numpy(),oh,oc,reset.numpy()]))
            h,c=encoder.step(tokens,prop,elapsed,h,c,reset)
            oh,oc=sess.run(None,feed)
            maximum=max(maximum,float(np.abs(oh-h.numpy()).max()),float(np.abs(oc-c.numpy()).max()))
    if maximum>1e-5: raise RuntimeError(f'Recurrent export parity failed: {maximum}')
    return dict(steps=64,reset_steps=[0,17,45],max_abs=maximum,tolerance=1e-5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','bank','encoder','planner','output'): p.add_argument('--'+name,required=True)
    args=p.parse_args()
    import mjlab
    import torch
    import onnx
    from .contracts import parse_contract,write_json
    from .control import file_hash
    from .models import GeometryMemory
    from .policy import HikingGeometryActor
    from ..policy import observation_template
    from ..env_cfg import build_agent_cfg
    from ..deploy.export import robot_contract
    from ..deploy.runtime import session
    torch.set_num_threads(2)
    data=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    metadata=data.get('infos',{}).get('hiking_perception',{})
    if metadata.get('mode')!='geometry': raise ValueError('Only RGB geometry policies can be exported; oracle is privileged')
    perception=Path(args.checkpoint).parent/'perception.pt'
    if file_hash(perception)!=metadata['checkpoint_sha256']: raise ValueError('Perception checkpoint mismatch')
    state=torch.load(perception,map_location='cpu',weights_only=False)
    g,m=parse_contract(state['contract']); encoder=GeometryMemory(g,m); encoder.load_state_dict(state['encoder'])
    bank=json.loads((Path(args.bank)/'bank.json').read_text()); cfg=build_agent_cfg(bank['config'])
    kwargs=copy.deepcopy(cfg.actor); kwargs.pop('class_name')
    obs=observation_template(batch=1); obs['velocity_command']=torch.zeros(1,8); obs['hiking_latent']=torch.zeros(1,m.hidden_dim)
    actor=HikingGeometryActor(obs=obs,obs_groups=cfg.obs_groups,obs_set='actor',output_dim=29,geometry_dim=m.hidden_dim,**kwargs).cpu().eval()
    actor.load_state_dict(data['actor_state_dict'],strict=True)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    memory_parity=export_memory(encoder,out/'memory.onnx')
    graph=make_graph(actor)
    torch.onnx.export(graph,graph.get_dummy_inputs(),str(out/'policy.onnx'),opset_version=18,dynamo=False,input_names=graph.input_names,output_names=graph.output_names)
    onnx.checker.check_model(str(out/'policy.onnx')); onnx.checker.check_model(str(out/'memory.onnx'))
    sess=session(out/'policy.onnx','cpu'); maximum=0.; rng=torch.Generator().manual_seed(491)
    with torch.no_grad():
        for _ in range(32):
            sample=obs.clone()
            for key in sample.keys(True,True): sample[key]=torch.randn(sample[key].shape,generator=rng)*.3
            feed={name:(sample['kv_tokens','img_tokens'] if name=='kv_tokens__img_tokens' else sample[name]).numpy() for name in graph.input_names}
            got=sess.run(['actions'],feed)[0]; expected=actor(sample,stochastic_output=False).numpy()
            maximum=max(maximum,float(np.abs(got-expected).max()))
    if maximum>1e-5: raise RuntimeError(f'Actor ONNX parity failed: {maximum}')
    for name,src in [('encoder.onnx',args.encoder),('planner.onnx',args.planner)]: shutil.copy2(src,out/name)
    enc=session(out/'encoder.onnx','cpu')
    if [(x.name,x.shape,x.type) for x in enc.get_inputs()] != [('image_bgr_hwc',[1,63,112,3],'tensor(uint8)')]: raise ValueError('Theia preprocessing contract mismatch')
    encoded=enc.run(None,{'image_bgr_hwc':np.zeros((1,63,112,3),np.uint8)})
    if [x.shape for x in encoded]!=[(1,21,192),(1,192)]: raise ValueError('Theia output mismatch')
    robot=robot_contract(args.bank)
    if bank['joint_names']!=robot['joint_names']: raise ValueError('Joint order mismatch')
    write_json(out/'bundle.json',dict(schema='vibe_grail_joint.sim2real.v1',kind='velocity',
        required_runtime='vibe_grail_joint.hiking_perception.runtime.HikingRuntime',hiking_perception=dict(metadata,memory_parity=memory_parity),
        checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=file_hash(args.checkpoint),
        files={name:file_hash(out/name) for name in ('policy.onnx','memory.onnx','encoder.onnx','planner.onnx')},
        robot=robot,camera=bank['config']['camera'],vision_revision=bank['config']['vision_revision'],
        inputs={x.name:x.shape for x in sess.get_inputs()},policy_parity=dict(cases=32,maximum_absolute_error=maximum,tolerance=1e-5),
        reference=dict(frames=10,stride_s=.1,anchor='pelvis',quaternion='wxyz'),
        history=dict(length=10,order='oldest_to_newest',layout='term_major',initialization='repeat_first'),
        requires_reference_motion=True,depth_input=False,hardware_validated=False,standing_start_stop_validated=False))
    print(json.dumps(dict(output=str(out),policy_max_abs=maximum,memory=memory_parity),indent=2))


if __name__=='__main__': main()
