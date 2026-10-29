"""Build a small, provenance-preserving replay bank entirely on the CPU.

This is a fixed capability-retention panel, not a new training dataset or an
estimate of corpus-wide success. Flat clips come from training; GRAIL clips
come from the existing asset-family held-out split. No simulator, torch, CUDA,
planner, resampling, FK or coordinate transformation is involved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from ..catalog import asset_family, is_validation, write_json
from ..kinematics import CHANNELS
from ..prepare import BANK_VERSION, validate_bank

FLAT_SLOTS = (
    ("sonic", "walk", r"walk_forward_amateur"),
    ("sonic", "clap", r"clap_enthusiastic"),
    ("sonic", "stretch", r"body_stretch"),
    ("sonic", "boxing", r"shadow_boxing"),
    ("sonic", "gesture", r"gesture"),
    ("sonic", "mirror", r"looking_in_the_mirror"),
    ("style100", "arms_above_head", r"^ArmsAboveHead/"),
    ("style100", "arms_folded", r"^ArmsFolded/"),
    ("style100", "arms_behind_back", r"^ArmsBehindBack/"),
    ("style100", "aeroplane", r"^Aeroplane/"),
    ("lafan", "dance", r"^dance"),
    ("real_pico", "balance", r"^balance"),
)
SELECTION = {
    "version": 1,
    "case_count": 24,
    "flat_count": 12,
    "grail_count": 12,
    "min_frames": 100,
    "min_arms_range_rms_deg": 3.0,
    "min_upper_max_range_deg": 10.0,
    "flat_rule": "One amplitude-median eligible window per named category; no source/id repeats.",
    "grail_rule": "Twelve equal-count max-height strata; amplitude-median eligible clip in each; "
                  "distinct source/id and asset family; original complete windows.",
    "threshold_meaning": "Sample definition only; no policy outcome or simulation success was used.",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _upper_indices(names):
    waist = [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    arms = [f"{side}_{joint}_joint" for side in ("left", "right") for joint in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw",
    )]
    if len(names) != 29 or len(set(names)) != 29 or any(name not in names for name in waist + arms):
        raise ValueError("Expected 29 named joints including the three waist and fourteen arm joints")
    return [names.index(name) for name in waist + arms], waist + arms


def upper_motion(joint_pos, joint_names, control_hz):
    """Measure real channel values; joint groups are resolved by name, not index."""
    indices, names = _upper_indices(joint_names)
    values = np.asarray(joint_pos[:, indices], dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Invalid upper-body reference samples")
    ranges = np.rad2deg(np.ptp(values, axis=0))
    return {
        "joint_names": names,
        "joint_range_deg": ranges.tolist(),
        "upper_range_rms_deg": float(np.sqrt(np.mean(ranges**2))),
        "arms_range_rms_deg": float(np.sqrt(np.mean(ranges[3:]**2))),
        "waist_range_rms_deg": float(np.sqrt(np.mean(ranges[:3]**2))),
        "upper_max_range_deg": float(ranges.max()),
        "upper_finite_difference_speed_rms_rad_s": float(
            np.sqrt(np.mean((np.diff(values, axis=0) * control_hz)**2))
        ),
    }


def _source(path, expected_split):
    path = Path(path).resolve()
    metadata = validate_bank(path)
    if metadata["split"] != expected_split or metadata.get("smoke_only", False):
        raise ValueError(f"Expected a full {expected_split} source bank: {path}")
    if metadata["config"]["control_hz"] != 50:
        raise ValueError("The source_frame_start_50hz contract requires 50 Hz banks")
    _upper_indices(metadata["joint_names"])
    return {
        "path": path,
        "metadata": metadata,
        "joint_pos": np.memmap(path / "joint_pos.bin", mode="r", dtype=np.float32,
                               shape=(metadata["total_frames"], 29)),
        "provenance": {
            "path": str(path), "split": metadata["split"], "identity": metadata["identity"],
            "bank_json_sha256": hashlib.sha256((path / "bank.json").read_bytes()).hexdigest(),
            "model_signature": metadata["model_signature"],
        },
    }


def _candidate(source, index, label):
    metadata = source["metadata"]
    clip = metadata["clips"][index]
    start, length = clip["offset"], clip["length"]
    if length < SELECTION["min_frames"]:
        return None
    motion = upper_motion(source["joint_pos"][start:start + length], metadata["joint_names"], 50)
    if (motion["arms_range_rms_deg"] < SELECTION["min_arms_range_rms_deg"]
            or motion["upper_max_range_deg"] < SELECTION["min_upper_max_range_deg"]):
        return None
    return {"source_bank": source, "index": index, "label": label, "upper_motion": motion}


def _median(candidates):
    def key(candidate):
        clip = candidate["source_bank"]["metadata"]["clips"][candidate["index"]]
        return (candidate["upper_motion"]["arms_range_rms_deg"], clip["id"],
                clip["source_frame_start_50hz"], candidate["index"])
    return sorted(candidates, key=key)[len(candidates) // 2]


def select_cases(flat, grail):
    """Select the same panel for identical banks without random sampling."""
    selected, seen = [], set()
    for source_name, label, pattern in FLAT_SLOTS:
        candidates = []
        for index, clip in enumerate(flat["metadata"]["clips"]):
            if (clip["source"] != source_name or (source_name, clip["id"]) in seen
                    or not re.search(pattern, clip["id"], re.IGNORECASE)):
                continue
            if clip["terrain_id"] != -1:
                raise ValueError("A flat source clip unexpectedly has paired terrain")
            candidate = _candidate(flat, index, label)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            raise ValueError(f"No sufficiently moving flat candidate for {source_name}/{label}")
        candidate = _median(candidates)
        clip = flat["metadata"]["clips"][candidate["index"]]
        seen.add((source_name, clip["id"]))
        selected.append(candidate)

    metadata = grail["metadata"]
    candidates = []
    for index, clip in enumerate(metadata["clips"]):
        if clip["source"] != "grail":
            continue
        if clip["terrain_id"] < 0:
            raise ValueError("A GRAIL clip is missing paired terrain")
        terrain = metadata["terrain_rows"][clip["terrain_id"]]
        family = asset_family(terrain["source_stem"])
        config = metadata["config"]
        if not is_validation(family, config["validation_fraction"], config["seed"]):
            raise ValueError(f"GRAIL source violates the declared held-out family split: {clip['id']}")
        candidate = _candidate(grail, index, "grail")
        if candidate is not None:
            candidate.update(family=family, height=float(terrain["max_height"]))
            candidates.append(candidate)
    candidates.sort(key=lambda candidate: (candidate["height"], candidate["family"], candidate["index"]))
    if len(candidates) < 12:
        raise ValueError("Need at least twelve eligible held-out GRAIL clips")
    families = set()
    for stratum in range(12):
        start, end = stratum * len(candidates) // 12, (stratum + 1) * len(candidates) // 12
        pool = [candidate for candidate in candidates[start:end]
                if candidate["family"] not in families
                and ("grail", metadata["clips"][candidate["index"]]["id"]) not in seen]
        if not pool:
            raise ValueError(f"No distinct GRAIL family in height stratum {stratum}; revise the panel explicitly")
        candidate = _median(pool)
        candidate["label"] = f"grail_height_stratum_{stratum:02d}"
        candidate["height_stratum"] = stratum
        candidate["height_stratum_range_m"] = [candidates[start]["height"], candidates[end - 1]["height"]]
        clip = metadata["clips"][candidate["index"]]
        seen.add(("grail", clip["id"]))
        families.add(candidate["family"])
        selected.append(candidate)
    return selected


def build_replay_bank(flat_bank, grail_bank, output):
    requested_output = Path(output).absolute()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite replay bank: {requested_output}")
    output = requested_output.resolve()
    stage = output.with_name(output.name + ".partial")
    if output.exists() or output.is_symlink() or stage.exists() or stage.is_symlink():
        raise FileExistsError(f"Refusing to overwrite replay bank or partial bank: {output}")
    flat, grail = _source(flat_bank, "train"), _source(grail_bank, "validation")
    for key in ("version", "model_signature", "joint_names", "body_names", "channels", "config"):
        if flat["metadata"][key] != grail["metadata"][key]:
            raise ValueError(f"Source bank mismatch: {key}")
    selected = select_cases(flat, grail)
    manifest = {
        "version": 1,
        "purpose": "Fixed replay-role retention panel; not online-W generalization or corpus-wide validation.",
        "control_hz": 50,
        "selection": copy.deepcopy(SELECTION),
        "source_banks": {"flat": flat["provenance"], "grail": grail["provenance"]},
        "flat_source_split": "train",
        "grail_source_split": "validation",
        "bank_split_compatibility_note": "bank.split=train enables the mixed-source sampler; only cases "
                                         "tagged grail_heldout are from held-out data.",
        "channels_copied_without_transform": True,
        "flat_root_xy_rebased_by_builder": False,
        "cases": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    try:
        mappings = {
            str(source["path"]): {
                name: np.memmap(source["path"] / f"{name}.bin", mode="r", dtype=np.float32,
                                shape=(source["metadata"]["total_frames"], *tail))
                for name, tail in CHANNELS.items()
            } for source in (flat, grail)
        }
        clips, terrains, offset = [], [], 0
        hashes = {name: hashlib.sha256() for name in CHANNELS}
        with ExitStack() as stack:
            handles = {name: stack.enter_context((stage / f"{name}.bin").open("xb")) for name in CHANNELS}
            for case_id, candidate in enumerate(selected):
                source = candidate["source_bank"]
                original = source["metadata"]["clips"][candidate["index"]]
                clip = copy.deepcopy(original)
                terrain = None
                if clip["source"] == "grail":
                    terrain = copy.deepcopy(source["metadata"]["terrain_rows"][clip["terrain_id"]])
                    terrain_path = Path(source["metadata"]["config"]["terrain_root"]) / terrain["terrain"]
                    if not terrain_path.is_file():
                        raise ValueError(f"Missing paired terrain asset: {terrain_path}")
                    clip["terrain_id"] = len(terrains)
                    terrains.append(terrain)
                clip["offset"] = offset
                clips.append(clip)
                start, end = original["offset"], original["offset"] + original["length"]
                arrays = mappings[str(source["path"])]
                case_hashes = {}
                for name in CHANNELS:
                    value = np.asarray(arrays[name][start:end])
                    if not np.isfinite(value).all():
                        raise ValueError(f"Non-finite selected {name}: {original['source']}/{original['id']}")
                    payload = value.tobytes(order="C")
                    handles[name].write(payload)
                    hashes[name].update(payload)
                    case_hashes[name] = hashlib.sha256(payload).hexdigest()
                root_xy = np.asarray(arrays["body_pos_w"][start, 0, :2]).tolist()
                case = {
                    "case_id": case_id, "bank_clip_index": case_id,
                    "subset": "grail_heldout" if terrain else "flat_preservation",
                    "label": candidate["label"], "source": original["source"], "id": original["id"],
                    "source_bank": str(source["path"]), "source_bank_identity": source["metadata"]["identity"],
                    "source_split": source["metadata"]["split"], "original_clip_index": candidate["index"],
                    "original_clip": copy.deepcopy(original),
                    "source_frame_start_50hz": original["source_frame_start_50hz"],
                    "source_frame_end_50hz_exclusive": original["source_frame_start_50hz"] + original["length"],
                    "length": original["length"], "duration_s": (original["length"] - 1) / 50,
                    "whole_source_bank_clip": True, "upper_motion": candidate["upper_motion"],
                    "terrain_id": clip["terrain_id"], "original_terrain_id": original["terrain_id"],
                    "terrain": terrain, "channel_sha256": case_hashes,
                    "source_first_root_xy_m": root_xy,
                    "flat_source_root_xy_is_zero": bool(np.allclose(root_xy, 0, atol=1e-6)) if terrain is None else None,
                    "flat_root_xy_rebased_by_builder": False,
                }
                if terrain:
                    case.update(asset_family=candidate["family"], height_stratum=candidate["height_stratum"],
                                height_stratum_range_m=candidate["height_stratum_range_m"],
                                terrain_path=str(terrain_path.resolve()),
                                terrain_sha256=hashlib.sha256(terrain_path.read_bytes()).hexdigest())
                manifest["cases"].append(case)
                offset += original["length"]
        counts = dict(Counter(clip["source"] for clip in clips))
        if counts != {"sonic": 6, "style100": 4, "lafan": 1, "real_pico": 1, "grail": 12}:
            raise ValueError(f"Unexpected panel composition: {counts}")
        manifest.update(total_frames=offset, channel_bytes=offset * sum(int(np.prod(t)) for t in CHANNELS.values()) * 4,
                        channel_sha256={name: value.hexdigest() for name, value in hashes.items()})
        metadata = copy.deepcopy(flat["metadata"])
        metadata.update(version=BANK_VERSION, identity=_digest(manifest), rank=0, world_size=1,
                        split="train", smoke_only=False, evaluation_only=True, total_frames=offset,
                        clips=clips, terrain_rows=terrains, source_motion_counts=counts, source_clip_counts=counts,
                        replay_manifest="manifest.json")
        manifest["bank_identity"] = metadata["identity"]
        write_json(stage / "bank.json", metadata)
        write_json(stage / "manifest.json", manifest)
        validate_bank(stage)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output appeared during construction: {output}")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return manifest


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-bank", type=Path, default=root / "data/full/rank_0")
    parser.add_argument("--grail-bank", type=Path, default=root / "data/validation/rank_0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_replay_bank(args.flat_bank, args.grail_bank, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(manifest["cases"]),
                      "total_frames": manifest["total_frames"], "channel_bytes": manifest["channel_bytes"],
                      "bank_identity": manifest["bank_identity"]}, indent=2))


if __name__ == "__main__":
    main()
