"""Functional invariants for the shared action path and training-only guidance."""
import torch
import pytest
from tensordict import TensorDict
from rsl_rl.modules.cross_attention import CrossAttentionExtractor
from vibe_grail_joint.vggt_distill.model import SpatialExtractor, position_encoding


def extractors():
    options = dict(token_dim=192, latent_dim=128, token_terms=['img_tokens'],
                   query_groups=['q_proprio', 'q_task_cmd', 'q_cls'],
                   query_dims={'q_proprio': 64, 'q_task_cmd': 6, 'q_cls': 192},
                   obs_normalization=False)
    torch.manual_seed(31)
    base = CrossAttentionExtractor(**options)
    student = SpatialExtractor(**options, patch_grid=(4, 7))
    missing, unexpected = student.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith('spatial_blocks.') for k in missing)
    return base, student


def test_identity_initialization_preserves_action_condition():
    base, student = extractors()
    inputs = TensorDict({'img_tokens': torch.randn(3, 28, 192)}, batch_size=[3])
    queries = [torch.randn(3, n) for n in (64, 6, 192)]
    torch.testing.assert_close(student(inputs, *queries), base(inputs, *queries), rtol=0, atol=0)


def test_spatial_loss_updates_shared_path_then_changes_policy_feature():
    _, student = extractors()
    tokens = torch.randn(2, 28, 192)
    inputs = TensorDict({'img_tokens': tokens}, batch_size=[2])
    queries = [torch.randn(2, n) for n in (64, 6, 192)]
    before = student(inputs, *queries).detach()
    target = torch.randn_like(tokens)
    loss = (1 - torch.nn.functional.cosine_similarity(student.spatial_features(tokens)[1], target, dim=-1)).mean()
    loss.backward()
    grads = [p.grad for p in student.spatial_blocks.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.square().sum()) for g in grads) > 0
    torch.optim.SGD(student.parameters(), lr=.5).step()
    after = student(inputs, *queries).detach()
    assert (before - after).abs().max() > 1e-6


def test_positions_preserve_two_dimensional_patch_order():
    p = position_encoding(4, 7, 2048, 'cpu')
    assert p.shape == (28, 2048)
    assert not torch.equal(p[0], p[1])
    assert not torch.equal(p[0], p[7])
    torch.testing.assert_close(p[0, :1024], p[7, :1024])
    torch.testing.assert_close(p[0, 1024:], p[1, 1024:])


def test_unrecognized_patch_shape_fails_loudly():
    import pytest
    _, student = extractors()
    with pytest.raises(ValueError, match='patch grid'):
        student.spatial_features(torch.zeros(2, 21, 192))


def test_evaluation_scenarios_are_ascending_and_complete():
    from vibe_grail_joint.vggt_distill.scenes import cases_and_tiles
    settings = {'planner_speeds': [0., .25, .4, .5, .7]}
    tiles, cases = cases_and_tiles(settings)
    assert len(cases) == 120
    assert all(t['kind'] == 'up' and all(h > 0 for h in t['jumps']) for t in tiles)
    assert sum(c['out_of_training_height'] for c in cases) == 24
    assert cases_and_tiles(settings) == (tiles, cases)
    assert cases_and_tiles(settings, irregular=True)[0] != tiles


def test_comparison_keeps_failed_trials_and_rejects_mismatch():
    import copy
    import pytest
    from vibe_grail_joint.vggt_distill.compare import compare_reports
    report = dict(ascent_only=True, checkpoint='example.pt', checkpoint_iteration=1, tiles=[{'kind': 'up'}],
                  initial_state={'root': [[0., 0., 0.]], 'joints': [[0.]]}, cases=[])
    for i, success in enumerate((True, False)):
        report['cases'].append(dict(case={'id': i, 'nominal_height_m': .2}, success=success,
            collision_free=False, foot_impulse_ns=2. + 8*i, shin_impulse_ns=0., speed_error_m_s=.1))
    compared = compare_reports(report, copy.deepcopy(report))
    assert compared['by_height'][0]['left']['mean_foot_impulse_ns'] == 6.
    assert compared['by_height'][0]['left']['success_rate'] == .5
    wrong = copy.deepcopy(report); wrong['cases'][0]['case']['id'] = 99
    with pytest.raises(ValueError, match='Cases'):
        compare_reports(report, wrong)
    for route in ('down', 'up_down'):
        route_report = copy.deepcopy(report)
        route_report.update(route=route, ascent_only=False, completion_mode='walking')
        matched = compare_reports(route_report, copy.deepcopy(route_report))
        assert matched['route'] == route and matched['by_height'][0]['left']['count'] == 2
        wrong = copy.deepcopy(route_report); wrong['completion_mode'] = 'strict'
        with pytest.raises(ValueError, match='Completion'):
            compare_reports(route_report, wrong)
        wrong = copy.deepcopy(route_report); wrong['route'] = 'up'
        with pytest.raises(ValueError, match='route'):
            compare_reports(route_report, wrong)


def test_review_routes_have_correct_landings_and_fit_physical_atlas():
    import numpy as np
    from vibe_grail_joint.vggt_distill.scenes import cases_and_tiles
    settings = {'planner_speeds': [0., .25, .4, .5, .7]}
    for route in ('up', 'down', 'up_down'):
        tiles, cases = cases_and_tiles(settings, heights=(.2, .25, .3), approaches=(0.,), route=route)
        for tile in tiles:
            levels = np.r_[tile['base'], tile['base'] + np.cumsum(tile['jumps'])]
            assert np.min(levels) > -1e-6
            assert tile['edges'][-1] < tile['goal'] < 12.
            assert len(tile['edges']) <= 16
            assert np.all(np.diff(tile['edges']) > 0)
            if route == 'down':
                assert levels[0] > 1 and abs(levels[-1]) < 1e-6
            elif route == 'up_down':
                assert abs(levels[0]) < 1e-6 and abs(levels[-1]) < 1e-6
                assert levels.max() > 1 and tile['start'] >= 2.4
                assert tile['goal'] - tile['edges'][-1] >= 2.
        assert all(c['out_of_training_direction'] == (route != 'up') for c in cases)


@pytest.mark.parametrize('config_name,height', [
    ('vggt_stairs_middle_20261027.json', 63),
    ('vggt_stairs_middle_rgb64_20261027.json', 64),
    ('vggt_stairs_hierarchical_rgb64_20261027.json', 64),
])
def test_corrected_task_preserves_original_data_terrain_rewards_and_camera(config_name, height):
    import json
    from pathlib import Path
    import mjlab
    from vibe_grail_joint.vggt_distill.task import env_config, task_fractions
    from vibe_grail_joint.velocity.quality_task import env_config as baseline
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root/'configs'/config_name).read_text())
    cfg = env_config(settings['bank_root']+'/rank_0', settings, 8000)
    old = baseline(settings['bank_root']+'/rank_0', settings, 8000)
    assert type(cfg.commands['motion']) is type(old.commands['motion'])
    assert cfg.commands['motion'].random_tiles == old.commands['motion'].random_tiles
    assert {t['kind'] for t in cfg.commands['motion'].random_tiles} == {'up', 'down', 'up_down', 'flat'}
    atlas = cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    assert atlas.rows == old.scene.terrain.terrain_generator.sub_terrains['paired_stairs'].rows
    assert len(atlas.rows) == 1450
    assert cfg.rewards.keys() == old.rewards.keys()
    for key in cfg.rewards:
        assert cfg.rewards[key].weight == old.rewards[key].weight
        assert cfg.rewards[key].func == old.rewards[key].func
    assert task_fractions(settings) == dict(online_ppo=.5, paired_grail_ppo=.2, flat_dagger_planner=.15, flat_dagger_replay=.15)
    camera = next(s for s in cfg.scene.sensors if s.name == 'head_cam')
    assert (camera.width, camera.height) == (112, height)


@pytest.mark.parametrize('height,width,grid', [(63,112,(3,7)), (64,111,(4,6)), (64,112,(3,7))])
def test_full_frame_distillation_rejects_implicit_crop_or_wrong_grid(height, width, grid):
    from vibe_grail_joint.vggt_distill.visual_contract import validate_patch_coverage
    with pytest.raises(ValueError, match='patch coverage|patch grid mismatch'):
        validate_patch_coverage(height, width, grid)


def test_fixed_configs_only_change_rgb_height_and_matching_grid():
    import copy
    import json
    from pathlib import Path
    from vibe_grail_joint.vggt_distill.visual_contract import validate_training_visual_contract
    root = Path(__file__).resolve().parents[1]
    for variant in ('middle', 'hierarchical'):
        original = json.loads((root/f'configs/vggt_stairs_{variant}_20261027.json').read_text())
        corrected = json.loads((root/f'configs/vggt_stairs_{variant}_rgb64_20261027.json').read_text())
        expected = copy.deepcopy(original)
        expected['camera']['height'] = 64
        expected['distillation']['patch_grid'] = [4, 7]
        assert corrected == expected
        assert validate_training_visual_contract(corrected)['patch_count'] == 28
        with pytest.raises(ValueError, match='Incomplete patch coverage'):
            validate_training_visual_contract(original)


def test_bottom_patch_row_changes_actual_frozen_theia_features():
    # Exercise the actual encoder, not a stand-in patch convolution. The pinned
    # local weights are already required for this project's training environment.
    from vibe_grail_joint.vision import FrozenTheiaFeature
    model = FrozenTheiaFeature._load('theia', 'theia-tiny-patch16-224-cddsv',
                                     'cpu', torch.float32).backbone.model
    with torch.inference_mode():
        for height in (63, 64):
            image = torch.zeros(1, 3, height, 112)
            baseline = model(pixel_values=image, interpolate_pos_encoding=True).last_hidden_state
            image[:, :, 48:, :] = 1.
            changed = model(pixel_values=image, interpolate_pos_encoding=True).last_hidden_state
            if height == 64:
                assert baseline.shape == (1, 29, 192)
                assert float((changed[:, 1:] - baseline[:, 1:]).abs().max()) > 1e-4
            else:
                assert baseline.shape == (1, 22, 192)
                torch.testing.assert_close(changed, baseline, rtol=0, atol=0)
