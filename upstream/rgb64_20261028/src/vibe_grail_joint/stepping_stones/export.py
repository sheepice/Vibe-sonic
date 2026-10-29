"""CPU export/parity check of all three variants; runtime needs no physics truth."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np

def make_graph(actor):
    import torch
    from rsl_rl.models.extractor_model import _OnnxExtractorSonicAdapterModel
    from rsl_rl.models._onnx_export import Port
    class Graph(_OnnxExtractorSonicAdapterModel):
        def _adapter_ports(self,model):
            ports=super()._adapter_ports(model)
            self.names=[p.name for p in ports]
            self.variant=model.stone_variant
            self.command_encoder=copy.deepcopy(model.command_encoder)
            self.geometry_encoder=copy.deepcopy(model.geometry_encoder)
            if self.variant=='moe':
                self.router=copy.deepcopy(model.router);self.experts=copy.deepcopy(model.experts)
            return ports+[Port('velocity_command',(8,),('velocity_command',)),Port('hiking_latent',(672,),('hiking_latent',))]
        def _adapter_latent(self,inputs):
            if self.variant=='geometry_only':
                base=torch.cat((self.adapter_normalizer(torch.cat(inputs[:self.num_plain_inputs],-1)),torch.zeros_like(inputs[-1][...,:128])),-1)
            else:base=super()._adapter_latent(inputs[:-2])
            base=base+self.command_encoder(inputs[-2])+self.geometry_encoder(inputs[-1])
            if self.variant=='moe':
                proprio=inputs[self.names.index('q_proprio')]
                x=torch.cat((inputs[-1],proprio,inputs[-2]),-1);x=x.sign()*torch.log1p(x.abs())
                weights=self.router(x).softmax(-1)
                base=base+(weights[...,None]*torch.stack([e(x) for e in self.experts],-2)).sum(-2)
            return base
    return Graph(actor,with_attn=False).cpu().eval()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--encoder',default='exports/20261018_sim2real_velocity3999/encoder.onnx')
    p.add_argument('--planner',default='/mnt/workspace/wjw/GRAIL/planner_sonic.onnx')
    a=p.parse_args()
    import mjlab,torch,onnx
    from torch import nn
    from ..policy import observation_template
    from .runtime import load_torch_control
    from ..deploy.export import sha256,robot_contract
    from ..deploy.runtime import session
    torch.set_num_threads(2)
    settings=json.loads(Path(a.config).read_text());obs=observation_template(batch=1)
    obs['velocity_command']=torch.zeros(1,8);obs['hiking_latent']=torch.zeros(1,672)
    actor,memory=load_torch_control(a.checkpoint,obs,settings,'cpu')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    graph=make_graph(actor)
    torch.onnx.export(graph,graph.get_dummy_inputs(),str(out/'policy.onnx'),opset_version=18,dynamo=False,input_names=graph.input_names,output_names=graph.output_names)
    onnx.checker.check_model(str(out/'policy.onnx'));sess=session(out/'policy.onnx','cpu')
    maximum=0.;rng=torch.Generator().manual_seed(619)
    with torch.no_grad():
        for i in range(32):
            sample=obs.clone()
            for k in sample.keys(True,True):sample[k]=torch.randn(sample[k].shape,generator=rng)*.3
            feed={x.name:(sample['kv_tokens','img_tokens'] if x.name=='kv_tokens__img_tokens' else sample[x.name]).numpy() for x in sess.get_inputs()}
            expected=actor(sample,stochastic_output=False).numpy();got=sess.run(['actions'],feed)[0]
            maximum=max(maximum,float(abs(expected-got).max()))
    if maximum>1e-5:raise RuntimeError(f'Actor parity {maximum}')
    class MemoryGraph(nn.Module):
        def __init__(self,model):super().__init__();self.model=model
        def forward(self,tokens,proprio,hidden,cell,reset):return self.model.step(tokens,proprio,hidden,cell,reset)
    encoder=memory.model.cpu().eval();h,c=encoder.initial(1,'cpu')
    dummy=(torch.zeros(1,21,192),torch.zeros(1,64),h,c,torch.ones(1,dtype=torch.bool))
    names=['tokens','proprio','hidden','cell','reset']
    torch.onnx.export(MemoryGraph(encoder),dummy,str(out/'memory.onnx'),opset_version=18,dynamo=False,input_names=names,output_names=['latent','next_hidden','next_cell'])
    onnx.checker.check_model(str(out/'memory.onnx'));ms=session(out/'memory.onnx','cpu');maximum_memory=0.
    oh,oc=h.numpy(),c.numpy()
    with torch.no_grad():
        for t in range(64):
            tok=torch.randn(1,21,192,generator=rng)*.3;prop=torch.randn(1,64,generator=rng)*.3;reset=torch.tensor([t in (0,19,43)])
            oz,oh,oc=ms.run(None,dict(tokens=tok.numpy(),proprio=prop.numpy(),hidden=oh,cell=oc,reset=reset.numpy()))
            z,h,c=encoder.step(tok,prop,h,c,reset)
            maximum_memory=max(maximum_memory,float(abs(z.numpy()-oz).max()),float(abs(h.numpy()-oh).max()),float(abs(c.numpy()-oc).max()))
    if maximum_memory>2e-5:raise RuntimeError(f'Memory parity {maximum_memory}')
    shutil.copy2(a.encoder,out/'encoder.onnx');shutil.copy2(a.planner,out/'planner.onnx')
    contract=robot_contract(settings['bank_root']+'/rank_0')
    bank=json.loads((Path(settings['bank_root'])/'rank_0/bank.json').read_text())
    manifest=dict(schema='vibe_grail_joint.sim2real.v1',kind='velocity',required_runtime='vibe_grail_joint.stepping_stones.runtime.StoneRuntime',
        variant=actor.stone_variant,checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=sha256(a.checkpoint),
        files={n:sha256(out/n) for n in ('policy.onnx','memory.onnx','encoder.onnx','planner.onnx')},
        robot=contract,camera=bank['config']['camera'],vision_revision=bank['config']['vision_revision'],
        inputs={x.name:x.shape for x in sess.get_inputs()},policy_parity=dict(cases=32,maximum_absolute_error=maximum),
        memory_parity=dict(steps=64,reset_steps=[0,19,43],maximum_absolute_error=maximum_memory),
        deploy_inputs=['RGB','proprioception','independent command','online SONIC reference','explicit hidden state'],
        depth_input=False,privileged_actor_inputs=False,hardware_validated=False,
        geometry_decoder_in_bundle=False,standing_start_stop_validated=False,
        deployment_targets=settings['stones'].get('deployment_targets', {'enabled':False}),
        reference=dict(frames=10,stride_s=.1,anchor='pelvis',quaternion='wxyz'),
        history=dict(length=10,order='oldest_to_newest',layout='term_major',initialization='repeat_first'))
    (out/'bundle.json').write_text(json.dumps(manifest,indent=2))
    from .runtime import StoneRuntime
    runtime=StoneRuntime(out,'cpu');runtime.warmup(2)
    print(json.dumps(dict(output=str(out),actor_parity=maximum,memory_parity=maximum_memory,runtime_warmup=True)),flush=True)
if __name__=='__main__':main()
