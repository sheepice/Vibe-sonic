"""Check/apply the optional pinned C++ runtime patch, with mandatory backups."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from .export import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkout", required=True)
    p.add_argument("--patch", required=True)
    p.add_argument("--apply", action="store_true", help="Default only verifies and displays the proposed diff")
    p.add_argument("--backup-root", help="Required with --apply; originals are copied here first")
    args = p.parse_args()
    checkout, patch = Path(args.checkout).resolve(), Path(args.patch).resolve()
    spec = json.loads(patch.with_suffix(".json").read_text())
    for relative, digest in spec["source_sha256"].items():
        if sha256(checkout / relative) != digest:
            raise ValueError(f"Source changed; left untouched: {relative}. Re-review the patch for this revision.")
    subprocess.run(["git", "-C", str(checkout), "apply", "--check", str(patch)], check=True)
    print(patch.read_text())
    if not args.apply:
        print("CHECK ONLY: no source modified")
        return
    if not args.backup_root:
        p.error("--apply requires --backup-root")
    backup = Path(args.backup_root).resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup.mkdir(parents=True, exist_ok=False)
    for relative in spec["source_sha256"]:
        dest = backup / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkout / relative, dest)
    write_json(backup / "receipt.json", dict(checkout=str(checkout), patch=str(patch), source_sha256=spec["source_sha256"]))
    subprocess.run(["git", "-C", str(checkout), "apply", str(patch)], check=True)
    print(f"Applied. Original bytes backed up at {backup}. Rebuild and run the official tests before control.")


if __name__ == "__main__":
    main()
