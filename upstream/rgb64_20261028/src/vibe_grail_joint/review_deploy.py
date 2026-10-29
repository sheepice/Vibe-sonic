"""ViBe native ONNX deployment components on our held-out stair environment.

Exports both the pinned visual encoder and trained actor. ViBe's OnnxAgent
assembles the manifest ports and drives world 1; world 0 runs the checkpoint.
Selected cases verify deployment, not an unbiased policy success estimate.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .gpu_guard import verify_guarded_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", type=int, default=6)
    args = parser.parse_args()
    verify_guarded_worker()
    import mjlab  # noqa: F401
    import numpy as np
    import onnxruntime as ort
    import torch
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx
    from vibe.encoders.zoo import _TheiaEnc
    from vibe.export.agent.onnx_agent import OnnxAgent
    from vibe.export.encoder.export_onnx import EncoderGraph, _parity_gate
    from vibe.export.manifest import build_encoder_manifest, build_manifest, duplicate_group_aliases

    from .catalog import write_json
    from .evaluation_env import EvaluationEnv
    from .policy import make_actor
    from .review_helpers import attach_observer, write_video
    from .stairs import build_stairs_env_cfg, validate_stairs_bank
    from .vision import FrozenTheiaFeature, THEIA_REVISION

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    bank = validate_stairs_bank(Path(args.bank))
    if bank["split"] != "validation" or bank["smoke_only"]:
        raise ValueError("Deployment checks require the full held-out bank")
    source = json.loads(Path(args.case_report).read_text())
    successes = sorted((r for r in source["episodes"] if r["ascent_success"]),
                       key=lambda r: r["reference_height_gain_m"])
    if not successes:
        raise ValueError("No successful ascents available for deployment parity cases")
    positions = np.linspace(0, len(successes) - 1, min(args.cases, len(successes)), dtype=int)
    selected = [successes[i]["clip_id"] for i in positions]
    indices = {c["id"]: i for i, c in enumerate(bank["clips"])}

    class PinnedTheia(_TheiaEnc):
        def _load(self):
            self._m = FrozenTheiaFeature._load("theia", bank["config"]["vision_encoder"],
                                             self.dev, self.dtype)
            self._vit = self._m.backbone.model
            self.patch = self._vit.config.patch_size

    encoder = PinnedTheia("theia-tiny", "theaiinstitute/" + bank["config"]["vision_encoder"],
                         "cpu", torch.float32)
    graph = EncoderGraph(encoder).eval()
    dummy = torch.zeros(1, 63, 112, 3, dtype=torch.uint8)
    encoder_path = destination / "theia-tiny.onnx"
    torch.onnx.export(graph, (dummy,), str(encoder_path), opset_version=18, dynamo=False,
                      input_names=["image_bgr_hwc"], output_names=["patch_tokens", "cls_token"])
    encoder_error = _parity_gate(graph, encoder_path, 63, 112, 8, 1e-4, "cpu")
    encoder_manifest = build_encoder_manifest(encoder, height=63, width=112, num_patches=21,
                                             token_dim=192, has_cls=True, compute_dtype="fp32", opset=18)
    encoder_manifest.update(revision=THEIA_REVISION, random_frame_parity_max_abs_error=encoder_error)
    attach_metadata_to_onnx(str(encoder_path), {"manifest": json.dumps(encoder_manifest)})
    write_json(destination / "theia-tiny.manifest.json", encoder_manifest)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    encoder_session = ort.InferenceSession(str(encoder_path), sess_options=options,
                                          providers=["CPUExecutionProvider"])

    cfg = build_stairs_env_cfg(args.bank, num_envs=2, play=True)
    cfg.seed = 42
    attach_observer(cfg)
    env = RslRlVecEnvWrapper(EvaluationEnv(cfg, device="cuda:0"))
    try:
        actor = make_actor(bank["config"], env.get_observations()).to("cuda:0").eval()
        actor.load_state_dict(torch.load(args.checkpoint, map_location="cuda:0",
                                        weights_only=False)["actor_state_dict"])
        reference = copy.deepcopy(actor).cpu().eval()
        exported = reference.as_onnx(with_attn=True).cpu().eval()
        aliases = duplicate_group_aliases(env.unwrapped, exported.layout)
        if aliases:
            exported.alias_ports(aliases)
        policy_path = destination / "model_14999.onnx"
        torch.onnx.export(exported, exported.get_dummy_inputs(), str(policy_path), opset_version=18,
                          dynamo=False, input_names=exported.input_names,
                          output_names=exported.output_names, dynamic_axes={})
        manifest = build_manifest(env.unwrapped, exported, task_id="ViBe-GRAIL-Stairs-Only",
                                  run_path=str(Path(args.checkpoint).resolve().parent),
                                  checkpoint=str(Path(args.checkpoint).resolve()), model_class=type(actor).__name__)
        # The upstream manifest recognizes its image_feature class by name. Our pinned
        # subclass has a different name; restore the same vision source tags explicitly.
        for port in manifest["inputs"]:
            for term in port["terms"]:
                if term["name"] in ("img_tokens", "img_cls"):
                    term["source"] = {"kind": "vision_encoder", "sensor": "head_cam",
                                      "model_name": bank["config"]["vision_encoder"],
                                      "revision": THEIA_REVISION, "model_dtype": "float32",
                                      "slice": "tokens" if term["name"] == "img_tokens" else "cls"}
        manifest.update(hardware_validated=False, requires_reference_motion=True,
                        camera=bank["config"]["camera"], evaluation_status="pending")
        attach_metadata_to_onnx(str(policy_path), {"manifest": json.dumps(manifest)})
        write_json(destination / "model_14999.manifest.json", manifest)
        agent = OnnxAgent(str(policy_path), provider="cpu")
        # Keep official port validation/assembly, bound CPU inference thread count.
        agent.session = ort.InferenceSession(str(policy_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        assert agent.joint_names == bank["joint_names"]
        command = env.unwrapped.command_manager.get_term("motion")
        reports = []
        policy_error = encoder_live_error = 0.0
        for clip_id in selected:
            index = indices[clip_id]
            command.forced_clip_ids = torch.tensor([index, index], device=env.device)
            env.unwrapped.record_terminals = False
            obs, _ = env.reset()
            env.unwrapped.record_terminals = True
            finished = [False, False]
            terminals = {}
            films = []
            head_films = []
            horizon = bank["clips"][index]["length"] + 60
            with torch.no_grad():
                for step in range(horizon):
                    if step % 2 == 0:
                        films.append(env.unwrapped.scene["review_camera"].data.rgb[1].cpu().numpy().copy())
                        head_films.append(env.unwrapped.scene["head_cam"].data.rgb[1].cpu().numpy().copy())
                    actions = actor(obs)
                    # Same-input gate: CPU checkpoint vs native ONNX actor; no device gap.
                    cpu_action = reference(obs[0:1].cpu()).numpy()
                    ort_action = agent.act(obs, 0)
                    policy_error = max(policy_error, float(np.abs(cpu_action - ort_action).max()))
                    if policy_error > 1e-5:
                        raise RuntimeError(f"Native actor ONNX parity failed: {policy_error}")
                    # World 1 uses BOTH exported artifacts, starting with raw camera RGB.
                    rgb = env.unwrapped.scene["head_cam"].data.rgb[1:2].cpu().numpy()
                    bgr = rgb[..., ::-1].copy()
                    dense, cls = encoder_session.run(None, {"image_bgr_hwc": bgr})
                    with torch.no_grad():
                        encoder_reference = graph(torch.from_numpy(bgr))
                    encoder_live_error = max(encoder_live_error, *(float(np.abs(a.numpy() - b).max())
                                                for a, b in zip(encoder_reference, (dense, cls))))
                    if encoder_live_error > 1e-4:
                        raise RuntimeError(f"Live encoder ONNX parity failed: {encoder_live_error}")
                    deploy_obs = obs.clone()
                    deploy_obs["kv_tokens", "img_tokens"][1:2] = torch.from_numpy(dense).to(env.device)
                    deploy_obs["q_cls"][1:2] = torch.from_numpy(cls).to(env.device)
                    actions[1] = torch.from_numpy(agent.act(deploy_obs, 1)[0]).to(env.device)
                    env.unwrapped.terminal_records = {}
                    obs, _, dones, _ = env.step(actions)
                    for world in (0, 1):
                        if bool(dones[world]) and not finished[world]:
                            finished[world] = True
                            terminals[str(world)] = {"steps": step + 1, **env.unwrapped.terminal_records[world]}
                    if all(finished):
                        break
            video = destination / f"{clip_id}_onnx_observer.mp4"
            head_video = destination / f"{clip_id}_onnx_rgb.mp4"
            stop = (terminals.get("1", {}).get("steps", horizon) + 1) // 2
            write_video(video, films[:stop])
            write_video(head_video, head_films[:stop])
            reports.append({"clip_id": clip_id, "torch": terminals.get("0", {"error": "horizon"}),
                            "onnx": terminals.get("1", {"error": "horizon"}),
                            "observer_video": str(video.resolve()), "rgb_video": str(head_video.resolve())})
            report = {"checkpoint": str(Path(args.checkpoint).resolve()), "bank_identity": bank["identity"],
                      "official_components": ["actor.as_onnx", "EncoderGraph", "build_manifest", "OnnxAgent"],
                      "policy_parity_max_abs_error": policy_error, "policy_parity_tolerance": 1e-5,
                      "encoder_random_parity_max_abs_error": encoder_error,
                      "encoder_live_parity_max_abs_error": encoder_live_error,
                      "raw_rgb_to_joint_actions_via_two_onnx_graphs": True,
                      "onnx_execution_provider": "CPUExecutionProvider", "cases": reports,
                      "selected_from_successful_validation_episodes": True,
                      "not_an_unbiased_success_rate": True, "hardware_validated": False}
            write_json(destination / "deployment_check.json", report)
            print(f"Deployment case {len(reports)}/{len(selected)}: {reports[-1]}", flush=True)
        manifest.update(evaluation_status="simulation_checked", hardware_validated=False,
                        simulation_check="deployment_check.json")
        write_json(destination / "model_14999.manifest.json", manifest)
    finally:
        env.close()


if __name__ == "__main__":
    main()
