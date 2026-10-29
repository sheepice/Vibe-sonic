"""Chunked, scene-disjoint, time-aligned supervision (no simulator dependency)."""
import hashlib
import json
from pathlib import Path
from collections import OrderedDict
import torch
from ..hiking_perception.contracts import parse_contract, scene_split, write_json


class ShardWriter:
    def __init__(self, directory, metadata, chunk_steps=128):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = dict(metadata, schema=1, shards=[], status='collecting',scene_key_version=metadata.get('scene_key_version',2))
        self.chunk_steps, self.rows = chunk_steps, []
        write_json(self.directory/'manifest.json', self.manifest)

    def add(self, row):
        self.rows.append({k: v.detach().cpu().clone() for k, v in row.items()})
        if len(self.rows) >= self.chunk_steps:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        data = {k: torch.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        name = f'shard_{len(self.manifest["shards"]):05d}.pt'
        path = self.directory/name
        torch.save(data, str(path)+'.tmp'); Path(str(path)+'.tmp').replace(path)
        self.manifest['shards'].append(dict(file=name, steps=len(self.rows), envs=data['valid'].shape[1],
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            valid_samples=int(data['valid'].sum()), surface_cells=int(data['surface_valid'].sum())))
        self.rows.clear()
        write_json(self.directory/'manifest.json', self.manifest)

    def close(self, **metrics):
        self.flush(); self.manifest.update(status='complete', **metrics)
        write_json(self.directory/'manifest.json', self.manifest)

