"""Chunked, scene-disjoint, time-aligned supervision (no simulator dependency)."""
import hashlib
import json
from pathlib import Path
from collections import OrderedDict
import torch
from .contracts import parse_contract, scene_split, write_json


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
            valid_samples=int(data['valid'].sum()), visible_cells=int(data['visible'].sum())))
        self.rows.clear()
        write_json(self.directory/'manifest.json', self.manifest)

    def close(self, **metrics):
        self.flush(); self.manifest.update(status='complete', **metrics)
        write_json(self.directory/'manifest.json', self.manifest)


class SequenceDataset:
    """Windows never span a scene change, reset, invalid transition or shard.

    The recurrent learner uses a burn-in prefix for each independent window.
    LRU caching bounds host memory; the corpus need not fit in RAM or VRAM.
    """
    def __init__(self, directories, length, split='train', stride=8, verify=True):
        self.length, self.windows, self.files, self.cache = length, [], [], OrderedDict()
        self.contract = None
        self.positive_windows = []
        signatures, scenes = [], set()
        for directory in directories:
            directory = Path(directory)
            manifest = json.loads((directory/'manifest.json').read_text())
            if manifest['status'] != 'complete':
                raise ValueError(f'Collection is incomplete: {directory}')
            if manifest.get('scene_key_version') != 2:
                raise ValueError(f'Legacy scene split: run hiking_perception.reindex on {directory} before fitting/validation')
            if self.contract is None:
                self.contract = manifest['contract']
            if manifest['contract'] != self.contract:
                raise ValueError('Incompatible geometry/feature contracts')
            keys = {int(k): v for k, v in manifest['scene_keys'].items()}
            signatures.append(hashlib.sha256(json.dumps(keys,sort_keys=True).encode()).hexdigest())
            for shard in manifest['shards']:
                path = directory/shard['file']
                if verify and hashlib.sha256(path.read_bytes()).hexdigest() != shard['sha256']:
                    raise ValueError(f'Corrupt dataset shard: {path}')
                signatures.append(shard['sha256'])
                file_id = len(self.files); self.files.append(path)
                data = torch.load(path, map_location='cpu', weights_only=True)
                for env in range(data['valid'].shape[1]):
                    for start in range(0, len(data['valid'])-length+1, stride):
                        sl = slice(start, start+length)
                        ids = data['scene_id'][sl, env]
                        if not torch.all(ids == ids[0]):
                            continue
                        key = keys[int(ids[0])]
                        if scene_split(key) != split:
                            continue
                        if not data['valid'][sl, env].all() or data['reset'][start+1:start+length, env].any():
                            continue
                        if 'contact' in data and data['contact'][sl, env].any():
                            self.positive_windows.append(len(self.windows))
                        self.windows.append((file_id, env, start)); scenes.add(key)
                del data
        if not self.windows:
            raise ValueError(f'No valid {split} windows of {length} steps; collect more independent scenes/longer shards')
        self.fingerprint = hashlib.sha256(''.join(sorted(signatures)).encode()).hexdigest()
        self.scenes = sorted(scenes)
        self.by_file = {}
        for index, (file_id, _, _) in enumerate(self.windows):
            self.by_file.setdefault(file_id, []).append(index)
        self.positive_by_file = {}
        for index in self.positive_windows:
            self.positive_by_file.setdefault(self.windows[index][0], []).append(index)
        parse_contract(self.contract)

    def _load(self, index):
        if index not in self.cache:
            self.cache[index] = torch.load(self.files[index], map_location='cpu', weights_only=True)
        self.cache.move_to_end(index)
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return self.cache[index]

    def sample(self, batch_size, generator, device, collision_fraction=0.):
        # Select a shard proportional to its valid-window count, then draw
        # windows within it. Each window retains the same marginal probability;
        # one disk read per minibatch replaces up to batch_size shard loads.
        if not 0 <= collision_fraction <= 1: raise ValueError('Invalid collision sample fraction')
        if collision_fraction and not self.positive_windows:
            raise ValueError('Collision-balanced sampling requires positive trajectories')
        first = (self.positive_windows[int(torch.randint(len(self.positive_windows), (), generator=generator))]
                 if collision_fraction else int(torch.randint(len(self.windows), (), generator=generator)))
        candidates = self.by_file[self.windows[first][0]]
        choices = [candidates[i] for i in torch.randint(len(candidates), (batch_size,), generator=generator).tolist()]
        positive_count = round(batch_size*collision_fraction)
        if positive_count:
            positive = self.positive_by_file[self.windows[first][0]]
            choices[:positive_count] = [positive[i] for i in torch.randint(len(positive), (positive_count,), generator=generator).tolist()]
        rows = []
        for choice in choices:
            file_id, env, start = self.windows[choice]
            data = self._load(file_id)
            row={k: v[start:start+self.length, env] for k, v in data.items()}
            if 'visible' in row:
                # Backward compatible: older corpora contain no historical views.
                row.setdefault('historical_visible',torch.zeros_like(row['visible']))
                row.setdefault('unknown',~row['visible'])
                row.setdefault('historical_seen_at_s',torch.full(row['visible'].shape,-1.))
            rows.append(row)
        batch = {k: torch.stack([r[k] for r in rows], 1).to(device) for k in rows[0]}
        for key in ('tokens', 'proprio', 'age', 'height', 'action', 'command', 'reference_twist', 'next_state', 'contact'):
            batch[key] = batch[key].float()
        batch['reset'][0] = True  # independent window; burn-in reconstructs memory
        if 'time_s' in batch:
            # Do not supervise memory of a view that predates this independent
            # training window: the encoder has never received that image.
            batch['historical_visible'] &= batch['historical_seen_at_s'] >= batch['time_s'][0][None,:,None]
        return batch
