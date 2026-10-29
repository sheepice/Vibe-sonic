"""Validate a completed two-rank diagnostic before full-size continuation."""
import argparse
import json
import math
from pathlib import Path

from .deployment_suite import digest, utc
from .front_geometry_loss import KIND


def validate(run):
    import torch
    torch.set_num_threads(2)
    run = Path(run).resolve()
    ranks = []
    for rank in (0, 1):
        status = json.loads((run/f'rank_{rank}_status.json').read_text())
        if (status['status'] != 'completed' or status['completed_updates'] != 100
                or status['world_size'] != 2 or status['num_envs'] != 80
                or not status['diagnostic']):
            raise ValueError('Require the complete 100-update, two-rank, 80-world diagnostic')
        if (status['privileged_input_invariance_error'] != 0
                or status['teacher_base_max_error'] != 0 or not status['teacher_frozen']
                or status['hiking_perception']['depth_input']):
            raise ValueError('Frozen teacher or RGB-only policy contract failed')
        if status['curriculum_resume']['curriculum_changed']:
            raise ValueError('Geometry comparison must preserve the curriculum')
        if status['settings']['gaussgym']['occupancy_objective']['kind'] != KIND:
            raise ValueError('Wrong geometry objective')
        rows = [json.loads(line) for line in (run/f'rank_{rank}_metrics.jsonl').read_text().splitlines()]
        if [r['completed_updates'] for r in rows] != list(range(1, 101)):
            raise ValueError('Missing or repeated updates')
        if any(not math.isfinite(float(v)) for r in rows for v in r['loss'].values()):
            raise ValueError('Nonfinite training loss')
        geometry = [r['loss'] for r in rows if 'GaussGym/loss' in r['loss']]
        if len(geometry) != 10:
            raise ValueError('Require all ten perception update events')
        if any(g['GaussGym/distributed_parameter_max_error'] != 0
               or g['GaussGym/front_valid_cells'] <= 0
               or not 0 < g['GaussGym/front_loss_mix'] <= .5 for g in geometry):
            raise ValueError('Geometry synchronization or local supervision failed')
        ranks.append(dict(rank=rank, status_sha256=digest(run/f'rank_{rank}_status.json'),
                          completed_updates=100, geometry_events=len(geometry),
                          first_geometry_updates=geometry[0]['GaussGym/online_updates'],
                          last_geometry_updates=geometry[-1]['GaussGym/online_updates'],
                          geometry_losses=[g['GaussGym/loss'] for g in geometry],
                          peak_device_gib=status['max_observed_device_used_gib']))
    last = status['absolute_completed_updates']-1
    path = run/f'model_{last}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    counts = [0, 0]
    def finite(value):
        if torch.is_tensor(value) and value.is_floating_point():
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite saved checkpoint tensor')
            counts[0] += 1; counts[1] += value.numel()
        elif isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                finite(v)
    finite(checkpoint)
    geometry = checkpoint['infos']['gaussgym']
    if geometry['settings']['occupancy_objective']['kind'] != KIND:
        raise ValueError('Checkpoint did not save the new objective')
    for rank in (0, 1):
        replay = run/checkpoint['infos']['dagger_replay']['pattern'].format(rank=rank)
        finite(torch.load(replay, map_location='cpu', weights_only=False))
    source = Path(status['initialization'])
    original = torch.load(source, map_location='cpu', weights_only=False)
    if geometry['online_updates'] != original['infos']['gaussgym']['online_updates']+40:
        raise ValueError('Perception optimizer update counter was not continued')
    return dict(verified_utc=utc(), passed=True, run=str(run), ranks=ranks,
                checkpoint=str(path), checkpoint_sha256=digest(path),
                finite_tensors=counts[0], finite_elements=counts[1],
                source_checkpoint=str(source), source_sha256=digest(source),
                production_must_resume_original_source=True,
                hardware_validated=False, locomotion_improvement_validated=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    result = validate(args.run)
    with Path(args.output).open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
