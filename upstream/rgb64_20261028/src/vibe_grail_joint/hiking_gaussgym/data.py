"""Memory-mapped sequences and explicit route/speed/approach validation strata."""
import collections
import hashlib
import json
from pathlib import Path
import torch
from ..hiking_perception.contracts import scene_key, scene_split

STAGES = ['approach_far','approach_near','on_stairs','after_stairs','flat']
FIELDS = ('tokens','proprio','height','surface_valid','base_lin_vel_b','base_ang_vel_b','valid','reset')


class Sequences:
    def __init__(self, directories, sequence=64, split='train', stride=16):
        self.sequence, self.split = sequence, split
        self.files, self.windows, self.metadata = [], [], []
        self.groups = collections.defaultdict(list)
        self.cache = collections.OrderedDict()
        self.contract = None
        scenes, signatures = set(), []
        for directory in directories:
            directory = Path(directory)
            manifest = json.loads((directory/'manifest.json').read_text())
            if manifest['status']!='complete' or manifest.get('scene_key_version')!=2:
                raise ValueError('Complete physical-geometry-v2 collection required')
            if self.contract is None: self.contract=manifest['contract']
            if manifest['contract']!=self.contract: raise ValueError('Mixed data contracts')
            keys={int(k):v for k,v in manifest['scene_keys'].items()}
            kinds={scene_key(tile):tile['kind'] for tile in manifest['tiles']}
            signatures.append(hashlib.sha256(json.dumps(keys,sort_keys=True).encode()).hexdigest())
            for shard in manifest['shards']:
                path=directory/shard['file']
                digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
                if digest.hexdigest()!=shard['sha256']: raise ValueError(f'Corrupt shard {path}')
                signatures.append(shard['sha256'])
                file_id=len(self.files); self.files.append(path)
                data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                for key in (*FIELDS,'scene_id','terrain_stage','target_speed'):
                    if key not in data: raise ValueError(f'Missing {key}: collect with V2 collector')
                if len(data['valid'])<sequence: continue
                starts=list(range(0,len(data['valid'])-sequence+1,stride))
                for env in range(data['valid'].shape[1]):
                    for start in starts:
                        sl=slice(start,start+sequence)
                        ids=data['scene_id'][sl,env]
                        canonical=keys[int(ids[0])]
                        if scene_split(canonical)!=split: continue
                        if not torch.all(ids==ids[0]) or not data['valid'][sl,env].all() or data['reset'][start+1:start+sequence,env].any(): continue
                        stage=int(data['terrain_stage'][start+sequence//2,env])
                        speed=round(float(data['target_speed'][start,env]),2)
                        group=(str(manifest['seed']),kinds[canonical],f'{speed:.2f}',STAGES[stage])
                        index=len(self.windows)
                        self.windows.append((file_id,env,start)); self.metadata.append((group,canonical))
                        self.groups[group].append(index); scenes.add(canonical)
        if not self.windows: raise ValueError(f'No {split} windows')
        self.fingerprint=hashlib.sha256(''.join(sorted(signatures)).encode()).hexdigest()
        self.scenes=sorted(scenes); self.group_keys=sorted(self.groups)
        self.group_weights=torch.tensor([1.5 if k[1] in ('down','up_down') else 1. for k in self.group_keys])

    def _load(self, index):
        if index not in self.cache:
            self.cache[index]=torch.load(self.files[index],map_location='cpu',weights_only=True,mmap=True)
        self.cache.move_to_end(index)
        while len(self.cache)>4: self.cache.popitem(last=False)
        return self.cache[index]

    def batch(self, indices, device):
        rows=[]
        for index in indices:
            file_id,env,start=self.windows[index]; data=self._load(file_id)
            rows.append({k:data[k][start:start+self.sequence,env] for k in FIELDS})
        batch={k:torch.stack([r[k] for r in rows],1).to(device) for k in FIELDS}
        for k in ('tokens','proprio','height','base_lin_vel_b','base_ang_vel_b'): batch[k]=batch[k].float()
        batch['reset'][0]=True
        return batch

    def sample(self, size, generator, device):
        # Select a balanced stratum, then shard-local windows to limit disk I/O.
        gi=int(torch.multinomial(self.group_weights,1,generator=generator))
        candidates=self.groups[self.group_keys[gi]]
        first=candidates[int(torch.randint(len(candidates),(),generator=generator))]
        file_id=self.windows[first][0]
        candidates=[i for i in candidates if self.windows[i][0]==file_id]
        indices=[candidates[i] for i in torch.randint(len(candidates),(size,),generator=generator).tolist()]
        return self.batch(indices,device)

    def fixed_plan(self, per_group=2, seed=138):
        if per_group<1: raise ValueError('Positive per-group validation size required')
        generator=torch.Generator().manual_seed(seed); plan=[]
        for key in self.group_keys:
            candidates=self.groups[key]
            order=torch.randperm(len(candidates),generator=generator).tolist()[:per_group]
            plan.extend(candidates[i] for i in order)
        return plan

    def plan_records(self, plan):
        return [dict(file=str(self.files[self.windows[i][0]]),env=self.windows[i][1],start=self.windows[i][2],
                     group=list(self.metadata[i][0]),scene=self.metadata[i][1]) for i in plan]
