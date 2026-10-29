"""Read existing datasets without changing them; split stairs by asset family."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def asset_family(stem: str) -> str:
    return re.sub(r"__\d+$", "", stem)


def is_validation(family: str, fraction: float, seed: int) -> bool:
    value = int(hashlib.sha256(f"{seed}:{family}".encode()).hexdigest()[:16], 16) / 2**64
    return value < fraction


def make_catalog(config: dict, split: str = "train") -> dict:
    if split not in ("train", "validation", "all"):
        raise ValueError(split)
    root = Path(config["terrain_root"])
    terrain = json.loads((root / "terrain_manifest.json").read_text())
    sources = {}
    for name, spec in {"grail": {"path": str(root), "weight": 1.0}, **config["flat_sources"]}.items():
        folder = Path(spec["path"])
        metadata = json.loads((folder / "manifest.json").read_text())
        if metadata["qpos_dim"] != 36 or len(metadata["qpos_names"]) != 36:
            raise ValueError(f"{name}: expected named 36-column G1 qpos")
        sources[name] = {**spec, "dt": metadata["timestep"], "qpos_names": metadata["qpos_names"]}
        sources[name]["manifest_sha256"] = hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
    rows = []
    for row in sorted(terrain["entries"], key=lambda r: r["id"]):
        family = asset_family(row["source_stem"])
        validation = is_validation(family, config["validation_fraction"], config["seed"])
        if split != "all" and validation != (split == "validation"):
            continue
        rows.append(
            {
                "source": "grail",
                "path": str(root / row["motion"]),
                "id": row["id"],
                "family": family,
                "terrain": row,
                "dt": 1.0 / row["fps"],
            }
        )
    # The flat corpus is retained in training; validation is held-out stairs.
    if split != "validation":
        for name, spec in config["flat_sources"].items():
            folder = Path(spec["path"])
            files = sorted((folder / "motions").rglob("*.npz"))
            expected = json.loads((folder / "manifest.json").read_text())["num_motions"]
            if len(files) != expected:
                raise ValueError(f"{name}: found {len(files)} motions, expected {expected}")
            for path in files:
                rows.append(
                    {
                        "source": name,
                        "path": str(path),
                        "id": str(path.relative_to(folder / "motions")),
                        "dt": sources[name]["dt"],
                    }
                )
    if not rows:
        raise ValueError("Empty dataset selection")
    return {
        "version": 1,
        "split": split,
        "sources": sources,
        "rows": rows,
        "counts": dict(Counter(r["source"] for r in rows)),
        "config": config,
    }


def partition_rows(rows: list[dict], rank: int, world_size: int) -> list[dict]:
    """Disjoint deterministic shards, balancing EACH source rather than a combined list."""
    if not 0 <= rank < world_size:
        raise ValueError("Invalid rank")
    result = []
    for source in sorted({r["source"] for r in rows}):
        subset = sorted((r for r in rows if r["source"] == source), key=lambda r: r["id"])
        result.extend(subset[rank::world_size])
    return result
