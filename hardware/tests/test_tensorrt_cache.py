"""Reject mismatched/unvalidated native engine caches before loading CUDA."""
import json

import pytest

from vibe_grail_joint.deploy.export import sha256
from vibe_grail_joint.deploy.tensorrt_runtime import validate_cache, SPATIAL_ANCHOR_SHAPES


def fixture(tmp_path):
    source,engine = tmp_path/'policy.onnx',tmp_path/'policy.plan'
    source.write_bytes(b'original ONNX')
    engine.write_bytes(b'GPU cache')
    required = ['encoder/patch_tokens','encoder/cls_token',
                'policy_same_inputs/actions','encoder_policy_chain/actions']
    report = dict(passed=True,cases=96,requested_cases=96,failure=None,
        tolerance_absolute=1e-5,required_outputs=required,
        max_absolute_error={k:1e-6 for k in required},
        engines={'policy':sha256(engine)},sources={'policy':sha256(source)})
    data = dict(schema='vibe.tensorrt_cache.v1',precision='float32_no_tf32',
        source_sha256=sha256(source),engine_sha256=sha256(engine),input_adapters={},
        numerically_validated=True)
    return source,engine,report,data


def save(engine,report,data):
    path = engine.parent/'parity.json'
    path.write_text(json.dumps(report))
    data['parity_sha256'] = sha256(path)
    engine.with_suffix('.json').write_text(json.dumps(data))


def test_validated_cache_accepts_matching_control_output_evidence(tmp_path):
    source,engine,report,data = fixture(tmp_path)
    save(engine,report,data)
    assert validate_cache(source,engine)['numerically_validated']


def test_rgb64_encoder_output_names_require_both_feature_checks(tmp_path):
    source,engine,report,data = fixture(tmp_path)
    report['required_outputs'][:2] = ['encoder/patches', 'encoder/cls']
    report['max_absolute_error'] = {k: 1e-6 for k in report['required_outputs']}
    save(engine,report,data)
    assert validate_cache(source,engine)['numerically_validated']
    report['required_outputs'].remove('encoder/cls')
    save(engine,report,data)
    with pytest.raises(ValueError, match='control-output'):
        validate_cache(source,engine)


def test_only_known_spatial_attention_output_anchors_are_accepted(tmp_path):
    source,engine,report,data = fixture(tmp_path)
    data['output_anchors'] = list(SPATIAL_ANCHOR_SHAPES)
    save(engine,report,data)
    assert validate_cache(source,engine)['numerically_validated']
    data['output_anchors'].pop()
    save(engine,report,data)
    with pytest.raises(ValueError, match='output anchors'):
        validate_cache(source,engine)


@pytest.mark.parametrize('kind',['source','engine','report'])
def test_changed_bytes_invalidate_cache(tmp_path,kind):
    source,engine,report,data = fixture(tmp_path)
    save(engine,report,data)
    path = {'source':source,'engine':engine,'report':engine.parent/'parity.json'}[kind]
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        validate_cache(source,engine)


@pytest.mark.parametrize('kind',['too_few','loose_tolerance','missing_output','bad_error','nan_error','incomplete'])
def test_pass_boolean_cannot_replace_numerical_evidence(tmp_path,kind):
    source,engine,report,data = fixture(tmp_path)
    if kind=='too_few':
        report.update(cases=1,requested_cases=1)
    elif kind=='loose_tolerance':
        report['tolerance_absolute']=1e-2
    elif kind=='missing_output':
        report['required_outputs']=[]
    elif kind=='incomplete':
        report['cases']=32
    else:
        report['max_absolute_error']['encoder_policy_chain/actions'] = .1 if kind=='bad_error' else float('nan')
    save(engine,report,data)
    with pytest.raises(ValueError,match='control-output'):
        validate_cache(source,engine)
