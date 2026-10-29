"""Offline first-LoRA features and patch occlusion sensitivity from saved inputs.

The first adapter consumes a separate condition stream. Official export merges
it as the final column block of decoder.0.weight, so its scaled output is still
recoverable independently of the base branch. Later adapters are merged by sum.
Original ONNX files and live simulation are never modified.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from rosbags.highlevel import AnyReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim.vendor import onnx_ml_pb2 as pb

OUT = ROOT / 'records/first_lora_features_20261029'
BUNDLE = ROOT / 'models/vggt_hierarchical_rgb64'
Z = '/extractors.0/attention/out_norm/LayerNormalization_output_0'
COND = '/Add_output_0'
JOINED = '/Concat_2_output_0'
PRE = '/decoder/decoder.0/Gemm_output_0'
POST = '/decoder/decoder.1/Mul_output_0'
DELTA = 'probe.first_adapter.delta'


def session(model):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2
    opt.inter_op_num_threads = 1
    opt.log_severity_level = 3
    return ort.InferenceSession(model, sess_options=opt, providers=['CPUExecutionProvider'])


def output(graph, name, width):
    value = graph.output.add()
    value.name = name
    value.type.tensor_type.elem_type = 1
    for dim in (1, width):
        value.type.tensor_type.shape.dim.add().dim_value = dim


def main():
    OUT.mkdir(exist_ok=True)
    binary = (BUNDLE/'policy.onnx').read_bytes()
    model = pb.ModelProto()
    model.ParseFromString(binary)
    manifest = json.loads((BUNDLE/'bundle.json').read_text())
    cfg = json.loads((BUNDLE/'agent_cfg.json').read_text())['actor']
    assert hashlib.sha256(binary).hexdigest() == manifest['files']['policy.onnx']
    assert cfg['rank'] == 16 and cfg['alpha'] == 1.0
    init = {v.name:v for v in model.graph.initializer}
    weight = np.frombuffer(init['decoder.0.weight'].raw_data, np.float32).reshape(init['decoder.0.weight'].dims)
    bias = np.frombuffer(init['decoder.0.bias'].raw_data, np.float32)
    assert weight.shape == (2048,1128)
    condition_width, base_width = 134, 994
    adapter_weight = weight[:,base_width:].copy()
    # This is the already-scaled product (alpha/rank) B A, not an arbitrary
    # low-rank factorization. Preserve the original merged action branch.
    tensor = model.graph.initializer.add()
    tensor.name = 'probe.first_adapter.delta_weight_T'
    tensor.data_type = 1
    tensor.dims.extend([condition_width,2048])
    tensor.raw_data = np.ascontiguousarray(adapter_weight.T).tobytes()
    node = model.graph.node.add()
    node.name = 'probe.first_adapter.output'
    node.op_type = 'MatMul'
    node.input.extend([COND,tensor.name])
    node.output.append(DELTA)
    for name,width in [(Z,128),(COND,134),(JOINED,1128),(PRE,2048),(POST,2048),(DELTA,2048)]:
        output(model.graph,name,width)
    probe = session(model.SerializeToString())
    original = session(str(BUNDLE/'policy.onnx'))
    encoder = session(str(BUNDLE/'encoder.onnx'))
    bag = ROOT/'records/g1_sim_compare_20261029/sim_15cm_30cm_04/bag'
    previous = {p['step_id']:p for p in json.loads((ROOT/'records/feature_layers_20261029/late_output_sensitivity.json').read_text())['probes']}
    records, arrays, images = [], {}, []
    with AnyReader([bag]) as reader:
        stamps = {}
        for c,stamp,raw in reader.messages(connections=[c for c in reader.connections if c.topic=='/vibe/policy/step']):
            step = json.loads(reader.deserialize(raw,c.msgtype).data)
            if step['step_id'] in (280,320,360):
                stamps[step['step_id']] = stamp
        assert len(stamps) == 3
        for step_id,stamp in sorted(stamps.items()):
            feed, bgr = {}, None
            for c,_,raw in reader.messages(start=stamp,stop=stamp+1):
                if c.topic.startswith('/vibe/policy/inputs/'):
                    m = reader.deserialize(raw,c.msgtype)
                    feed[c.topic.split('/')[-1]] = np.asarray(m.data,np.float32).reshape([d.size for d in m.layout.dim])
                elif c.topic == '/vibe/camera/policy/image':
                    m = reader.deserialize(raw,c.msgtype)
                    bgr = np.asarray(m.data).reshape(64,112,3)
            assert bgr is not None
            names = ['actions',Z,COND,JOINED,PRE,POST,DELTA]
            action,z,cond,joined,pre,post,delta = probe.run(names,feed)
            action_error = float(np.max(abs(action-original.run(['actions'],feed)[0])))
            assert action_error == 0, action_error
            assert np.array_equal(joined[:,base_width:],cond)
            manual_delta = cond @ adapter_weight.T
            base = joined[:,:base_width] @ weight[:,:base_width].T + bias
            reconstructed = base + delta
            delta_error = float(np.max(abs(delta-manual_delta)))
            fusion_error = float(np.max(abs(pre-reconstructed)))
            np.testing.assert_allclose(delta,manual_delta,atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(pre,reconstructed,atol=2e-5,rtol=2e-5)
            sensitivities = {'layernorm':[], 'first_lora_delta':[], 'first_fused_silu':[]}
            masked_delta,masked_post = [],[]
            for patch in range(28):
                y,x = divmod(patch,7)
                perturbed = bgr.copy()
                perturbed[y*16:(y+1)*16,x*16:(x+1)*16] = bgr.mean((0,1)).astype(np.uint8)
                tokens,cls = encoder.run(None,{'image_bgr_hwc':perturbed[None]})
                altered = dict(feed,kv_tokens__img_tokens=tokens,q_cls=cls)
                zi,di,hi,ai = probe.run([Z,DELTA,POST,'actions'],altered)
                for key,value,baseline in [('layernorm',zi,z),('first_lora_delta',di,delta),('first_fused_silu',hi,post)]:
                    sensitivities[key].append(float(np.sqrt(np.mean((value-baseline)**2))))
                masked_delta.append(di[0]);masked_post.append(hi[0])
                altered_error = float(np.max(abs(ai-original.run(['actions'],altered)[0])))
                assert altered_error == 0, altered_error
            old_error = float(np.max(abs(np.asarray(sensitivities['layernorm'])-previous[step_id]['after_layernorm_change_rmse'])))
            assert old_error < 1e-6,old_error
            arrays.update({f'step{step_id}_{key}':value[0] for key,value in
                           [('z',z),('condition',cond),('first_lora_delta',delta),('first_fused_pre',pre),('first_fused_silu',post),('first_base',base)]})
            arrays[f'step{step_id}_masked_delta'] = np.stack(masked_delta)
            arrays[f'step{step_id}_masked_fused_silu'] = np.stack(masked_post)
            rgb = bgr[...,::-1].copy()
            arrays[f'step{step_id}_rgb'] = rgb
            images.append(rgb)
            record = dict(step_id=step_id,time_s=step_id*.02,sensitivity_rmse=sensitivities,
                          action_error_after_extra_taps=action_error,first_delta_reconstruction_error=delta_error,
                          first_base_plus_delta_reconstruction_error=fusion_error,old_layernorm_rmse_error=old_error)
            records.append(record)
            print(json.dumps({k:v for k,v in record.items() if k!='sensitivity_rmse'}),flush=True)
    np.savez_compressed(OUT/'features.npz',**arrays)
    palette = np.asarray([[22,26,90],[22,95,220],[22,211,192],[255,231,60],[240,45,35]])
    canvas = Image.new('RGB',(1280,750),(20,27,34))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    titles = ['Policy RGB','LayerNorm z sensitivity','First LoRA delta sensitivity','First fused + SiLU sensitivity']
    for row,(record,rgb) in enumerate(zip(records,images)):
        y = row*250
        draw.text((5,y+3),f"SIM {record['time_s']:.1f} s | Independent 16x16 masks; nonvisual inputs fixed | Relative color per panel",font=font,fill='white')
        base = np.asarray(Image.fromarray(rgb).resize((320,183),Image.Resampling.BILINEAR))
        keys = [None,'layernorm','first_lora_delta','first_fused_silu']
        for col,(title,key) in enumerate(zip(titles,keys)):
            draw.text((320*col+5,y+26),title,font=font,fill='white')
            if key is None:
                im = Image.fromarray(base)
            else:
                weights = np.asarray(record['sensitivity_rmse'][key],np.float32).reshape(4,7)
                high = float(weights.max())
                values = np.asarray(Image.fromarray(weights/max(high,1e-12)).resize((320,183),Image.Resampling.BILINEAR))
                heat = np.stack([np.interp(values,np.linspace(0,1,5),palette[:,j]) for j in range(3)],axis=-1)
                im = Image.fromarray(np.round(base*.45+heat*.55).astype(np.uint8))
                draw.text((320*col+5,y+233),f'RMSE 0 .. {high:.6f}',font=font,fill='#bbccd8')
            canvas.paste(im,(320*col,y+47))
    canvas.save(OUT/'first_lora_sensitivity.png')
    report = dict(policy_sha256=hashlib.sha256(binary).hexdigest(),original_policy_unchanged=(BUNDLE/'policy.onnx').read_bytes()==binary,
                  layernorm_shape=[1,128],condition_shape=[1,134],first_lora_output_shape=[1,2048],
                  first_fused_silu_shape=[1,2048],rank=16,alpha=1.0,
                  condition='concat(normalized augmentation[6], visual z[128]) + velocity command encoder[134]',
                  first_adapter='delta = condition @ merged_decoder_weight[:,994:].T; the column block already includes alpha/rank',
                  first_fused='SiLU(base_decoder_linear(base_input) + delta)',
                  method='For each patch, replace only that 16x16 input region by image channel mean, recompute encoder patch and CLS; all original nonvisual policy inputs fixed. Plot RMSE at the selected output.',
                  limitations='2048 features are channels, not spatial positions. Image overlays are perturbation sensitivity, not attention weights. The actual 16-D internal low-rank coordinates are not recovered from merged weights.',
                  sources=['https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/modules/mlp_with_adapter.py#L169-L186',
                           'https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/models/_onnx_export.py#L51-L94'],
                  probes=records)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
