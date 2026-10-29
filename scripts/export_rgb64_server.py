"""Export the RGB64 distilled actor on the training host, using CPU only.

Run from the upstream project with its .venv. No training sources are patched.
The standard extractor exporter drops subclass spatial blocks; this wrapper
explicitly includes all three trained blocks before the original cross attention.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    import mjlab  # register installed task/plugin entry points before ORCS imports
    import torch
    from torch import nn
    import onnx
    import onnxruntime as ort
    from vibe_grail_joint.policy import observation_template
    from vibe_grail_joint.vggt_distill.model import DistilledActor, position_encoding
    from vibe_grail_joint.vggt_distill.visual_contract import validate_training_visual_contract
    from vibe_grail_joint.deploy.export import make_export_graph, robot_contract
    from vibe_grail_joint.vision import FrozenTheiaFeature, THEIA_REVISION
    from vibe.core.mdp.observations import image_feature
    torch.set_num_threads(2)
    checkpoint = Path(a.checkpoint).resolve()
    run = checkpoint.parent
    settings = json.loads((run/'config.json').read_text())
    agent = json.loads((run/'agent_cfg.json').read_text())
    visual = validate_training_visual_contract(settings)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    metadata = state.get('infos', {}).get('vggt_distillation')
    if not metadata or metadata['settings'] != settings['distillation'] or metadata['camera'] != settings['camera']:
        raise ValueError('Checkpoint and saved visual/distillation configuration disagree')
    obs = observation_template(batch=1)
    obs['kv_tokens', 'img_tokens'] = torch.zeros(1, visual['patch_count'], 192)
    obs['velocity_command'] = torch.zeros(1, 8)
    acfg = copy.deepcopy(agent['actor'])
    if acfg.pop('class_name') != 'vibe_grail_joint.vggt_distill.model.DistilledActor':
        raise ValueError('Unexpected actor class')
    actor = DistilledActor(obs=obs, obs_groups=agent['obs_groups'], obs_set='actor', output_dim=29, **acfg).cpu().eval()
    actor.load_state_dict(state['actor_state_dict'], strict=True)
    with torch.inference_mode():
        before = actor(obs, stochastic_output=False)
    del actor.alignment_heads, actor.relation_head, actor.progress_head
    with torch.inference_mode():
        if not torch.equal(before, actor(obs, stochastic_output=False)):
            raise RuntimeError('Removing training-only heads changed actions')

    class SpatialGraph(nn.Module):
        def __init__(self, extractor, attention):
            super().__init__()
            self.blocks = copy.deepcopy(extractor.spatial_blocks)
            self.attention = attention
            self.register_buffer('positions', position_encoding(*extractor.patch_grid, extractor.token_dim, 'cpu'))

        def forward(self, tokens, *queries):
            for block in self.blocks:
                tokens = block(tokens, self.positions)
            return self.attention(tokens, *queries)

    graph = make_export_graph(actor, True)
    if len(graph.extractors) != 1 or actor.spatial_extractor.token_terms != ['img_tokens']:
        raise ValueError('Unsupported extractor layout')
    graph.extractors[0] = SpatialGraph(actor.spatial_extractor, graph.extractors[0])
    graph.eval()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    policy_path = out/'policy.unchecked.onnx'
    torch.onnx.export(graph, graph.get_dummy_inputs(), str(policy_path), opset_version=18,
        dynamo=False, input_names=graph.input_names, output_names=graph.output_names)
    onnx.checker.check_model(str(policy_path))

    def session(path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])

    policy = session(policy_path)
    rng = torch.Generator().manual_seed(20261028)
    maximum = 0.
    with torch.inference_mode():
        for i in range(64):
            sample = obs.clone()
            for key in sample.keys(True, True):
                sample[key] = torch.randn(sample[key].shape, generator=rng)*.3
            sample['velocity_command'] = torch.tensor([[(-.1,0.,.15,.25,.4,.7)[i%6],0.,0.,0.,0.,1.,0.,0.]])
            feed = {n:(sample['kv_tokens','img_tokens'] if n=='kv_tokens__img_tokens' else sample[n]).numpy()
                    for n in graph.input_names}
            expected = actor(sample, stochastic_output=False).numpy()
            actual = policy.run(['actions'], feed)[0]
            maximum = max(maximum, float(abs(expected-actual).max()))
    if maximum > 1e-5:
        raise RuntimeError(f'Policy parity failed: {maximum}')
    policy_path.rename(out/'policy.onnx')
    print(json.dumps(dict(stage='policy_exported', max_abs_error=maximum)), flush=True)

    vision = FrozenTheiaFeature._load('theia','theia-tiny-patch16-224-cddsv','cpu',torch.float32)
    mean, std = image_feature._NORM['imagenet']

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = vision.backbone.model
            self.register_buffer('mean', torch.tensor(mean)[None,:,None,None])
            self.register_buffer('std', torch.tensor(std)[None,:,None,None])

        def forward(self, bgr):
            rgb = bgr.flip(-1).permute(0,3,1,2).float()/255.
            hidden = self.backbone(pixel_values=(rgb-self.mean)/self.std,
                interpolate_pos_encoding=True).last_hidden_state
            return hidden[:,1:], hidden[:,0]

    encoder = Encoder().cpu().eval()
    h,w = settings['camera']['height'],settings['camera']['width']
    enc_path = out/'encoder.unchecked.onnx'
    torch.onnx.export(encoder, (torch.zeros(1,h,w,3,dtype=torch.uint8),), str(enc_path),
        opset_version=18, dynamo=False, input_names=['image_bgr_hwc'], output_names=['patches','cls'])
    onnx.checker.check_model(str(enc_path))
    encoded = session(enc_path)
    enc_max = full_max = 0.
    with torch.inference_mode():
        for i in range(16):
            image = torch.randint(0,256,(1,h,w,3),dtype=torch.uint8,generator=rng)
            if i == 0: image.zero_()
            if i == 1: image.fill_(255)
            expected = encoder(image)
            actual = encoded.run(None, {'image_bgr_hwc':image.numpy()})
            for x,y in zip(expected,actual):
                enc_max = max(enc_max,float(abs(x.numpy()-y).max()))
            sample = obs.clone()
            sample['kv_tokens','img_tokens'], sample['q_cls'] = expected
            sample['velocity_command'] = torch.tensor([[.25,0.,0.,0.,0.,1.,0.,0.]])
            feed = {n:(sample['kv_tokens','img_tokens'] if n=='kv_tokens__img_tokens' else sample[n]).numpy()
                    for n in graph.input_names}
            feed['kv_tokens__img_tokens'],feed['q_cls'] = actual
            full_max = max(full_max,float(abs(actor(sample,stochastic_output=False).numpy()-policy.run(['actions'],feed)[0]).max()))
    if enc_max > 1e-4 or full_max > 1e-5:
        raise RuntimeError(f'RGB pipeline parity failed: encoder={enc_max}, action={full_max}')
    enc_path.rename(out/'encoder.onnx')
    shutil.copy2(settings['planner'], out/'planner.onnx')
    bank_path = Path(settings['bank_root'])/'rank_0'
    bank = json.loads((bank_path/'bank.json').read_text())
    contract = robot_contract(bank_path)
    if contract['joint_names'] != bank['joint_names']:
        raise RuntimeError('Joint order mismatch')
    camera = copy.deepcopy(bank['config']['camera'])
    camera.update(width=w,height=h)
    manifest = dict(schema='vibe_grail_joint.sim2real.v1',kind='velocity',
        model_family='vggt_distilled',variant=settings['distillation']['variant'],
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),training_iteration=int(state['iter']),
        files={n:digest(out/n) for n in ['policy.onnx','encoder.onnx','planner.onnx']},
        robot=contract,camera=camera,vision_revision=THEIA_REVISION,visual_contract=visual,
        inputs={x.name:x.shape for x in policy.get_inputs()},
        encoder_inputs={x.name:x.shape for x in encoded.get_inputs()},
        policy_parity=dict(cases=64,maximum_absolute_error=maximum,tolerance=1e-5),
        encoder_parity=dict(cases=16,maximum_absolute_error=enc_max,tolerance=1e-4),
        rgb_to_action_parity=dict(cases=16,maximum_absolute_error=full_max,tolerance=1e-5),
        spatial_blocks=3,teacher_deployment_required=False,auxiliary_heads_in_export=False,
        reference=dict(frames=10,stride_s=.1,anchor='pelvis',quaternion='wxyz'),
        history=dict(length=10,order='oldest_to_newest',layout='term_major',initialization='repeat_first'),
        requires_reference_motion=True,depth_input=False,hardware_validated=False,
        standing_start_stop_validated=False,
        runtime=dict(python='3.11',torch=torch.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__))
    (out/'bundle.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ['config.json','agent_cfg.json','task_contract.json']:
        shutil.copy2(run/name,out/name)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    main()
