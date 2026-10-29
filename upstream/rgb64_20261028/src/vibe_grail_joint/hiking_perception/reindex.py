"""Migrate legacy geometry hashes without modifying or duplicating trajectory shards."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from .contracts import scene_key,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(); source=Path(args.data).resolve();out=Path(args.output).resolve()
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest.get('status')!='complete': raise ValueError('Collection is incomplete')
    if manifest.get('scene_key_version')==2: raise ValueError('Already uses physical-geometry scene hashes')
    if 'tiles' in manifest:
        tiles=manifest['tiles']
    else:
        # Only migrate the known original normal-profile collector. Verify every
        # legacy hash before accepting reconstructed profiles; no guessed remap.
        from ..velocity.normal_stairs_geometry import normal_tiles
        s=manifest['settings'];n=s['normal_stairs']
        tiles=normal_tiles(manifest['seed'],s['profile_levels'],s['profile_variants'],n['minimum_tread_m'],n['maximum_tread_m'])
    fields=('edges','jumps','base','n','height','tread','start','plateau')
    mapping={hashlib.sha256(json.dumps({k:t[k] for k in fields if k in t},sort_keys=True).encode()).hexdigest():scene_key(t) for t in tiles}
    if not set(manifest['scene_keys'].values())<=set(mapping):
        raise ValueError('Cannot exactly reconstruct old geometry; provide original tiles rather than guessing')
    result=copy.deepcopy(manifest)
    result.update(scene_key_version=2,tiles=tiles,reindexed_from=str(source),
        scene_keys={k:mapping[v] for k,v in manifest['scene_keys'].items()},
        split_rule='Physical profile only; same flats share one hash. Prefix modulo 10: 0 test, 1 val, else train')
    out.mkdir(parents=True,exist_ok=False)
    for shard in manifest['shards']:
        target=source/shard['file']
        if hashlib.sha256(target.read_bytes()).hexdigest()!=shard['sha256']: raise ValueError('Corrupt source shard')
        (out/shard['file']).symlink_to(target)
    write_json(out/'manifest.json',result)
    print(json.dumps(dict(output=str(out),old_scenes=len(set(manifest['scene_keys'].values())),new_scenes=len(set(result['scene_keys'].values())),shards_are_symlinks=True)))


if __name__=='__main__': main()
