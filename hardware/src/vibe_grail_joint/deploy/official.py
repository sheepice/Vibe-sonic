"""Fetch the four immutable upstream WALK artifacts; never relabel local weights.

The lock is supplied by the official vibe_ws checkout. Only the GRAIL policy
pair and shared Theia pair are downloaded, with size and SHA-256 verification.
"""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request

from .export import sha256, write_json


def download(lock_path, output):
    lock = json.loads(Path(lock_path).read_text())
    root = Path(output)
    selected = {k: v for k, v in lock["files"].items()
                if k.startswith("models/walk/") or k.startswith("encoders/")}
    if len(selected) != 4:
        raise ValueError("Expected the walk policy/manifest and Theia encoder/manifest")
    for relative, record in selected.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = f"https://huggingface.co/{record['repo_id']}/resolve/{record['revision']}/{record['path']}"
            temporary = path.with_suffix(path.suffix + ".partial")
            request = urllib.request.Request(url, headers={"User-Agent": "vibe-grail-sim2real/1"})
            with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            if temporary.stat().st_size != record["bytes"] or sha256(temporary) != record["sha256"]:
                raise ValueError(f"Downloaded artifact failed integrity check: {relative}")
            temporary.rename(path)
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Existing artifact differs; left untouched: {path}")
        print(f"Verified {path}", flush=True)
    write_json(root / "upstream_receipt.json", dict(source_lock=str(Path(lock_path).resolve()), files=selected,
                                                  physical_robot_validated_here=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    download(args.lock, args.output)


if __name__ == "__main__":
    main()
