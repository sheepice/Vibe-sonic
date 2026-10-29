"""CPU-only, resumable FK cache and bounded-memory reference banks for each rank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from .catalog import make_catalog, partition_rows, write_json
from .kinematics import CHANNELS, convert_motion, windows

BANK_VERSION = 2


def build_model():
    # mjlab must precede ORCS/ViBe because of its entry-point task discovery.
    import mjlab  # noqa: F401
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mocke.mdp.joint_maps import G1_TRACKED_BODY_NAMES

    model = get_g1_robot_cfg().spec_fn().compile()
    return model, G1_TRACKED_BODY_NAMES


def model_signature(model, body_names) -> str:
    import mujoco

    digest = hashlib.sha256()
    digest.update(f"v1:{mujoco.__version__}:{body_names}".encode())
    for name in ("body_parentid", "body_pos", "body_quat", "jnt_type", "jnt_axis", "jnt_pos", "qpos0"):
        digest.update(np.asarray(getattr(model, name)).tobytes())
    return digest.hexdigest()


def cached_motion(row, source, model, body_names, hz, signature, cache_dir: Path):
    path = Path(row["path"])
    stat = path.stat()
    key = hashlib.sha256(
        json.dumps(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "dt": row["dt"],
                "hz": hz,
                "names": source["qpos_names"],
                "model": signature,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    target = cache_dir / key[:2] / f"{key}.npz"
    if not target.exists():
        channels = convert_motion(row, source, model, body_names, hz)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file, **channels)
        temporary.replace(target)
    with np.load(target, allow_pickle=False) as data:
        return {name: data[name] for name in CHANNELS}


def prepare_bank(catalog: dict, output: Path, rank: int, world_size: int, cache: Path):
    rows = partition_rows(catalog["rows"], rank, world_size)
    if not rows:
        raise ValueError(f"Empty rank {rank}")
    identity = hashlib.sha256(
        json.dumps(
            {"version": BANK_VERSION, "catalog": catalog, "rank": rank, "world_size": world_size},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    final = output / f"rank_{rank}"
    if (final / "bank.json").exists():
        metadata = json.loads((final / "bank.json").read_text())
        if metadata["identity"] != identity:
            raise ValueError(f"Existing bank has different inputs: {final}; use a new output directory")
        validate_bank(final)
        return metadata
    # A partial bank is never considered ready. Only our .partial directory is rewritten.
    stage = output / f"rank_{rank}.partial"
    stage.mkdir(parents=True, exist_ok=True)
    model, body_names = build_model()
    signature = model_signature(model, body_names)
    hz = catalog["config"]["control_hz"]
    handles = {name: (stage / f"{name}.bin").open("wb") for name in CHANNELS}
    clips, terrain_rows, offset = [], [], 0
    try:
        for index, row in enumerate(rows):
            values = cached_motion(
                row, catalog["sources"][row["source"]], model, body_names, hz, signature, cache
            )
            length = len(values["joint_pos"])
            terrain_id = -1
            if row["source"] == "grail":
                terrain_id = len(terrain_rows)
                terrain_rows.append(row["terrain"])
                spans = [(0, length)]
            else:
                spans = windows(length, int(catalog["config"]["flat_window_seconds"] * hz) + 1)
            for start, end in spans:
                for name in CHANNELS:
                    value = np.array(values[name][start:end], dtype=np.float32, copy=True)
                    if terrain_id >= 0 and name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                        value[-1] = 0.0  # the terminal stair pose is a one-second stationary hold
                    if name == "body_pos_w" and terrain_id < 0:
                        value[..., :2] -= value[0, 0, :2].copy()
                    if not np.isfinite(value).all():
                        raise ValueError(f"Non-finite {name} in {row['path']}")
                    value.tofile(handles[name])
                clips.append(
                    {
                        "source": row["source"],
                        "id": row["id"],
                        "offset": offset,
                        "length": end - start,
                        "source_frame_start_50hz": start,
                        "terrain_id": terrain_id,
                    }
                )
                offset += end - start
            if index % 100 == 0:
                print(f"[prepare rank={rank}] {index + 1}/{len(rows)} files, {offset} frames", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    import mujoco

    joints = [
        model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    metadata = {
        "version": BANK_VERSION,
        "identity": identity,
        "model_signature": signature,
        "rank": rank,
        "world_size": world_size,
        "split": catalog["split"],
        "smoke_only": catalog.get("smoke_only", False),
        "total_frames": offset,
        "channels": CHANNELS,
        "clips": clips,
        "terrain_rows": terrain_rows,
        "body_names": body_names,
        "joint_names": joints,
        "source_motion_counts": dict(Counter(r["source"] for r in rows)),
        "source_clip_counts": dict(Counter(c["source"] for c in clips)),
        "config": catalog["config"],
    }
    write_json(stage / "bank.json", metadata)
    validate_bank(stage)
    stage.rename(final)
    return metadata


def validate_bank(path: Path):
    metadata = json.loads((path / "bank.json").read_text())
    if metadata["version"] != BANK_VERSION:
        raise ValueError("Reference bank format changed; prepare a new bank directory")
    for name, tail in CHANNELS.items():
        expected = metadata["total_frames"] * int(np.prod(tail)) * 4
        if (path / f"{name}.bin").stat().st_size != expected:
            raise ValueError(f"Incomplete reference bank: {path}/{name}")
    offset = 0
    for clip in metadata["clips"]:
        if clip["offset"] != offset or clip["length"] < 2:
            raise ValueError("Invalid clip boundaries")
        offset += clip["length"]
        if clip["source"] == "grail" and metadata["terrain_rows"][clip["terrain_id"]]["id"] != clip["id"]:
            raise ValueError("Motion/terrain pairing mismatch")
    if offset != metadata["total_frames"]:
        raise ValueError("Frame count mismatch")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/joint.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="data/fk_cache")
    parser.add_argument("--split", choices=("train", "validation", "all"), default="train")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument(
        "--limit-per-source", type=int, default=0, help="CPU smoke data only; never a full run"
    )
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    catalog = make_catalog(config, args.split)
    world_size = args.world_size or config["world_size"]
    if args.limit_per_source:
        catalog["rows"] = [
            row
            for source in catalog["counts"]
            for row in [r for r in catalog["rows"] if r["source"] == source][: args.limit_per_source]
        ]
        catalog["smoke_only"] = True
    output = Path(args.output)
    write_json(output / "catalog.json", catalog)
    print(catalog["counts"], flush=True)
    if not args.catalog_only:
        for rank in [args.rank] if args.rank is not None else range(world_size):
            prepare_bank(catalog, output, rank, world_size, Path(args.cache))


if __name__ == "__main__":
    main()
