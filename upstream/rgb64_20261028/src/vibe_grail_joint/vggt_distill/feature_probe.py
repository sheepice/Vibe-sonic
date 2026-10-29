"""Read-only RGB, token, attention and image-patch intervention diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reference-report', required=True)
    args = p.parse_args()
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from .evaluate import load_actor
    from .scenes import evaluation_env
    from ..vision import FrozenTheiaFeature
    from vibe.core.mdp.observations import image_feature

    torch.set_num_threads(2)
    torch.manual_seed(260027)
    settings = json.loads(Path(args.config).read_text())
    reference = json.loads(Path(args.reference_report).read_text())
    assert reference['checkpoint_sha256'] == hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'request.json').write_text(json.dumps(vars(args), indent=2))
    case = reference['cases'][0]['case']
    env, settings, cases = evaluation_env(settings, seed=reference['args']['seed'], duration=60,
        heights=(case['nominal_height_m'],), speeds=(case['speed'],), approaches=(0.,),
        record_case=0, route=case['route'])
    samples = []
    action_errors = []
    try:
        obs, _ = env.reset()
        raw = env.unwrapped
        actor, iteration = load_actor(settings, obs, args.checkpoint, raw)
        ext = actor.spatial_extractor
        h, w = ext.patch_grid
        assert (h, w) == (4, 7)
        initial = dict(root=raw.scene['robot'].data.root_link_pos_w.cpu().tolist(),
                       joints=raw.scene['robot'].data.joint_pos.cpu().tolist())
        assert initial == reference['initial_state']
        scale = torch.as_tensor(raw.action_manager.get_term('joint_pos').scale, device=env.device).reshape(-1)
        assert scale.numel() in (1, 29)
        encoder = FrozenTheiaFeature._load('theia', 'theia-tiny-patch16-224-cddsv', env.device, torch.float16)
        mn, sd = image_feature._NORM['imagenet']
        mean = torch.tensor(mn, device=env.device, dtype=torch.float16)[None, :, None, None]
        std = torch.tensor(sd, device=env.device, dtype=torch.float16)[None, :, None, None]
        def encode(rgb):
            x = rgb.permute(0, 3, 1, 2).to(torch.float16) / 255.
            z = encoder.backbone.model(pixel_values=(x-mean)/std,
                interpolate_pos_encoding=True).last_hidden_state.float()
            return z[:, 1:], z[:, 0]
        previous = np.load(Path(args.reference_report).parent / 'trace.npz')['action']
        with torch.inference_mode():
            for step in range(reference['cases'][0]['steps']):
                action = actor(obs, stochastic_output=False)
                action_errors.append(float((action - torch.as_tensor(previous[step], device=env.device)).abs().max()))
                if step % 10 == 0:
                    rgb = raw.scene['head_cam'].data.rgb.clone()
                    features = [obs['kv_tokens']['img_tokens'], *ext.spatial_features(obs['kv_tokens']['img_tokens'])]
                    attention = ext.last_attn.clone()
                    assert attention.shape[-1] == 28
                    original_tokens, original_cls = encode(rgb)
                    encoding_error = max(float((original_tokens-obs['kv_tokens']['img_tokens']).abs().max()),
                                         float((original_cls-obs['q_cls']).abs().max()))
                    assert encoding_error < 1e-5, encoding_error
                    # Mask pixels BEFORE Theia, re-encoding both dense and CLS features.
                    # It is an intervention at the current state, not a closed-loop rollout.
                    # Include an unmasked control in the SAME batch to separate
                    # intervention effects from batch-dependent FP16 numerics.
                    altered = rgb.expand(29, -1, -1, -1).clone()
                    fill = rgb.float().mean((1, 2), keepdim=True).to(rgb.dtype)
                    for patch in range(28):
                        y, x = divmod(patch, 7)
                        altered[patch+1, y*16:(y+1)*16, x*16:(x+1)*16] = fill[0]
                    tokens, cls = encode(altered)
                    changed = obs.expand(29).clone()
                    changed['kv_tokens']['img_tokens'] = tokens
                    changed['q_cls'] = cls
                    altered_action = actor(changed, stochastic_output=False)
                    delta = ((altered_action[1:] - altered_action[:1])*scale).square().mean(-1).sqrt()
                    control_delta = float(((altered_action[:1]-action)*scale).square().mean().sqrt())
                    repeat = actor(obs, stochastic_output=False)
                    assert torch.equal(action, repeat), 'Diagnostics changed the policy output'
                    assert all(torch.isfinite(x).all() for x in [*features, attention, delta])
                    samples.append(dict(rgb=rgb[0].cpu().numpy(),
                        overview=raw.scene['review_camera'].data.rgb[0].cpu().numpy().copy(),
                        features=torch.stack(features, 1)[0].cpu().numpy(),
                        attention=attention[0].cpu().numpy(), action_delta=delta.cpu().numpy(),
                        time_s=step*raw.step_dt, encoding_error=encoding_error,
                        unmasked_batch_control_rmse_rad=control_delta))
                obs, _, done, _ = env.step(action)
                assert not done.any()
        query_labels = ext.query_labels
        pairs = actor.alignment_pairs
    finally:
        env.close()
    arrays = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
    np.savez_compressed(out / 'features.npz', **arrays)
    # This is a fresh simulation rollout. Matching initial pose does not imply
    # bit-identical GPU physics trajectories; do not claim it is the old video.
    print(json.dumps({'capture_complete': len(samples), 'previous_rollout_action_max_difference': max(action_errors)}), flush=True)

    from .teacher import GeometryTeacher
    teacher = GeometryTeacher(settings['distillation'], 'cuda:0')
    target = teacher(torch.from_numpy(arrays['rgb'])).float()
    teacher.assert_frozen()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['actor_state_dict']
    errors = []
    with torch.inference_mode():
        for i, (student_level, teacher_level) in enumerate(pairs):
            head = torch.nn.Sequential(torch.nn.LayerNorm(192), torch.nn.Linear(192, 512),
                                       torch.nn.GELU(), torch.nn.Linear(512, 2048))
            prefix = f'alignment_heads.{i}.'
            head.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
            prediction = head(torch.from_numpy(arrays['features'][:, student_level+1]))
            errors.append((1-F.cosine_similarity(prediction, target[:, teacher_level], dim=-1)).numpy())
    arrays['alignment_error'] = np.stack(errors, 1)
    np.savez_compressed(out / 'features.npz', **arrays)
    attn = arrays['attention']
    entropy = -(attn*np.log(np.maximum(attn, 1e-12))).sum(-1)/np.log(28)
    report = dict(variant=settings['distillation']['variant'], iteration=iteration,
        rgb_shape=[64,112,3], patch_grid=[4,7], samples=len(samples), query_labels=query_labels,
        normalized_attention_entropy_mean=entropy.mean(0).tolist(),
        attention_bottom_row_mass_mean=attn.reshape(len(attn),-1,4,7)[:,:,-1].sum(-1).mean(0).tolist(),
        image_patch_mask_action_target_rmse_rad_mean=float(arrays['action_delta'].mean()),
        image_patch_mask_action_target_rmse_rad_max=float(arrays['action_delta'].max()),
        alignment_pairs=pairs, alignment_error_mean=arrays['alignment_error'].mean((0,2)).tolist(),
        previous_rollout_action_max_difference=max(action_errors), initial_state_matches_reference=True,
        same_state_action_unchanged_by_diagnostics=True,
        unmasked_batch_control_rmse_rad_max=float(arrays['unmasked_batch_control_rmse_rad'].max()),
        rgb_reencode_max_error=float(arrays['encoding_error'].max()),
        checkpoint_sha256=reference['checkpoint_sha256'], reference_report=str(Path(args.reference_report).resolve()),
        note='Attention is not causal attribution. Mask probes replace one 16x16 RGB patch with the image mean and recompute dense+CLS; they measure instantaneous action sensitivity, not task success. PCA colors have no semantic meaning. Teacher targets use the same low-resolution RGB.')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    render(out, arrays, report)


def render(out, a, report):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio.v2 as iio
    # One PCA basis and fixed color scale across all frames and four student stages.
    x = a['features'].reshape(-1, 192).astype(np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    mean = x.mean(0)
    _, _, vh = np.linalg.svd(x-mean, full_matrices=False)
    z = (x-mean) @ vh[:3].T
    lo, hi = np.percentile(z, [1,99], axis=0)
    colors = np.clip((z-lo)/np.maximum(hi-lo,1e-12),0,1).reshape(len(a['rgb']),4,4,7,3)
    np.savez_compressed(out/'pca_basis.npz', mean=mean, basis=vh[:3], low=lo, high=hi)
    attention_max = max(.08, float(a['attention'].max()))
    delta_max = max(1e-6, float(a['action_delta'].max()))
    error_max = max(.01, float(a['alignment_error'].max()))
    selected = {min(len(a['rgb'])-1, round(t/.2)) for t in [1.0,2.0,3.0,4.0,5.0]}
    fig, axes = plt.subplots(3,4,figsize=(16,10),dpi=100)
    with iio.get_writer(out/'feature_diagnostics.mp4',fps=5,codec='libx264',quality=8,macro_block_size=1) as video:
        for i in range(len(a['rgb'])):
            for ax in axes.flat:
                ax.clear(); ax.set_xticks([]); ax.set_yticks([])
            axes[0,0].imshow(a['overview'][i]); axes[0,0].set_title('Fixed scene view')
            axes[0,1].imshow(a['rgb'][i],interpolation='nearest'); axes[0,1].set_title('Actual RGB 112x64 | patch16 grid')
            for xx in range(0,113,16):axes[0,1].axvline(xx-.5,color='cyan',lw=.6)
            for yy in range(0,65,16):axes[0,1].axhline(yy-.5,color='cyan',lw=.6)
            axes[0,2].imshow(colors[i,0],interpolation='nearest');axes[0,2].set_title('Theia tokens | joint PCA color')
            axes[0,3].imshow(colors[i,3],interpolation='nearest');axes[0,3].set_title('Spatial block 3 | same PCA basis')
            for j,label in enumerate(report['query_labels']):
                ax=axes[1,j];ax.imshow(a['attention'][i,j].reshape(4,7),cmap='magma',vmin=0,vmax=attention_max,interpolation='nearest')
                ax.set_title(f'Policy attention: {label}')
            axes[1,3].imshow(a['action_delta'][i].reshape(4,7),cmap='inferno',vmin=0,vmax=delta_max,interpolation='nearest')
            axes[1,3].set_title('RGB-patch mask: action sensitivity')
            axes[2,0].imshow(colors[i,1],interpolation='nearest');axes[2,0].set_title('Spatial block 1 | same PCA basis')
            axes[2,1].imshow(colors[i,2],interpolation='nearest');axes[2,1].set_title('Spatial block 2 | same PCA basis')
            axes[2,2].imshow(a['alignment_error'][i].mean(0).reshape(4,7),cmap='viridis',vmin=0,vmax=error_max,interpolation='nearest')
            axes[2,2].set_title('VGGT alignment: 1-cosine (mean heads)')
            axes[2,3].axis('off')
            axes[2,3].text(0,1, f"Fixed scales across this video\nAttention: 0 to {attention_max:.3f}\nAction target RMSE: 0 to {delta_max:.4f} rad\nAlignment error: 0 to {error_max:.3f}\n\nPCA color is not semantic class.\nAttention is not causal attribution.\nMasking measures one-state sensitivity.\nNearest-neighbor display: 4 x 7 cells.", va='top',fontsize=10)
            fig.suptitle(f"{report['variant']} | model_{report['iteration']} | 20cm up | 0.7m/s | t={a['time_s'][i]:.2f}s", fontsize=16)
            fig.tight_layout(rect=(0,0,1,.96));fig.canvas.draw()
            video.append_data(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy())
            if i in selected:fig.savefig(out/f"frame_{a['time_s'][i]:04.1f}s.png",dpi=120)
    plt.close(fig)
    print(json.dumps({'output':str(out),'report':report}),flush=True)


if __name__ == '__main__':
    main()
