"""Read-only validation of the completed dual-GPU crossing migration diagnostic."""
import argparse
import json
import math
from pathlib import Path

from .deployment_suite import digest, utc


def validate(run):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    torch.set_num_threads(2)
    run = Path(run).resolve()
    ranks = []
    for rank in (0, 1):
        status = json.loads((run/f'rank_{rank}_status.json').read_text())
        if (status['status'] != 'completed' or status['completed_updates'] != 100
                or status['num_envs'] != 80 or status['world_size'] != 2):
            raise ValueError('Require a completed two-rank 100-update diagnostic')
        if (status['privileged_input_invariance_error'] != 0 or status['teacher_base_max_error'] != 0
                or not status['teacher_frozen'] or status['hiking_perception']['depth_input']):
            raise ValueError('RGB-only/frozen SONIC contract failed')
        if not status['curriculum_resume'].get('gap_crossing_changed'):
            raise ValueError('Explicit crossing migration was not recorded')
        rows = [json.loads(line) for line in (run/f'rank_{rank}_metrics.jsonl').read_text().splitlines()]
        if [r['completed_updates'] for r in rows] != list(range(1, 101)):
            raise ValueError('Missing or repeated updates')
        if any(not math.isfinite(float(v)) for r in rows for v in r['loss'].values()):
            raise ValueError('Nonfinite losses')
        geom = [r['loss'] for r in rows if 'GaussGym/loss' in r['loss']]
        if len(geom) != 10 or any(g['GaussGym/distributed_parameter_max_error'] != 0 for g in geom):
            raise ValueError('Missing perception updates or distributed parameter mismatch')
        ranks.append(dict(rank=rank, completed_updates=100, geometry_events=len(geom),
                          peak_device_gib=status['max_observed_device_used_gib'],
                          status_sha256=digest(run/f'rank_{rank}_status.json')))
    events = EventAccumulator(str(run), size_guidance={'scalars':0}).Reload()
    tags = events.Tags()['scalars']
    recorded = {}
    for suffix in ('bridge_world_fraction','valid_crossings'):
        matches = [t for t in tags if t.endswith('GapCrossing/'+suffix)]
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous contact diagnostics')
        values = events.Scalars(matches[0])
        if not values or any(not math.isfinite(x.value) for x in values) or max(x.value for x in values) <= 0:
            raise ValueError('Both bridge and true crossing events must be exercised')
        recorded[suffix] = dict(tag=matches[0], positive_updates=sum(x.value>0 for x in values),
                                maximum_rollout_step_mean=max(x.value for x in values))
    path = run/f'model_{status["absolute_completed_updates"]-1}.pt'
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    source = Path(status['initialization'])
    original = torch.load(source,map_location='cpu',weights_only=False)
    if checkpoint['critic_state_dict']['mlp.0.weight'].shape != original['critic_state_dict']['mlp.0.weight'].shape:
        raise ValueError('Critic observation dimension changed')
    course = checkpoint['infos']['stepping_stones']['curriculum']
    if not torch.equal(course['frontier'][1:], torch.zeros_like(course['frontier'][1:])):
        raise ValueError('Sparse course did not restart from level zero')
    if checkpoint['infos']['gaussgym']['online_updates'] != original['infos']['gaussgym']['online_updates']+40:
        raise ValueError('Geometry optimizer update counter not continued')
    count = [0,0]
    def check(value):
        if torch.is_tensor(value) and value.is_floating_point():
            if not torch.isfinite(value).all(): raise ValueError('Nonfinite checkpoint')
            count[0]+=1; count[1]+=value.numel()
        elif isinstance(value,dict):
            for v in value.values():check(v)
        elif isinstance(value,(list,tuple)):
            for v in value:check(v)
    check(checkpoint)
    for rank in (0,1):
        check(torch.load(run/checkpoint['infos']['dagger_replay']['pattern'].format(rank=rank),
                         map_location='cpu',weights_only=False))
    return dict(utc=utc(),passed=True,run=str(run),ranks=ranks,contact_diagnostics=recorded,
        checkpoint=str(path),checkpoint_sha256=digest(path),source_checkpoint=str(source),
        source_sha256=digest(source),critic_input_dim=checkpoint['critic_state_dict']['mlp.0.weight'].shape[1],
        frontier=course['frontier'].tolist(),finite_tensors=count[0],finite_elements=count[1],
        production_must_resume_original_source=True,locomotion_improvement_validated=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=validate(a.run)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
