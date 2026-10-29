"""Read complete-rollout start/stop counts without changing a running trainer.

The rank JSON contains the final control step of each rollout. RSL-RL writes
TensorBoard means over every control step; the count tags are present on every
step, including zeros. Multiplying those means by the rollout length recovers
rank-zero completed-episode counts. These are training data, not deployment
success rates, and must not be extrapolated to the other rank or terrain levels.
"""
import argparse
import json
import math
from pathlib import Path


PREFIX = 'Stones/StartStop/'


def recover_counts(ended, successful, rollout_steps):
    """Recover integer episode counts; reject incomplete or ambiguous events."""
    if not isinstance(rollout_steps, int) or isinstance(rollout_steps, bool) or rollout_steps < 1:
        raise ValueError('Positive integer rollout length required')
    if not ended or set(ended) != set(successful):
        raise ValueError('Both count tags must cover exactly the same updates')
    steps = sorted(ended)
    if steps != list(range(steps[0], steps[-1]+1)):
        raise ValueError('Missing updates in TensorBoard count series')
    rows = []
    for step in steps:
        counts = []
        for value in (ended[step], successful[step]):
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid episode count mean')
            total = value*rollout_steps
            integer = round(total)
            if abs(total-integer) > 1e-3:
                raise ValueError('Count is not a complete-rollout mean')
            counts.append(integer)
        if counts[1] > counts[0]:
            raise ValueError('Successes exceed completed episodes')
        rows.append(dict(update=step, ended=counts[0], successful=counts[1]))
    return rows


def aggregate(rows):
    ended = sum(r['ended'] for r in rows)
    successful = sum(r['successful'] for r in rows)
    return dict(first_update=rows[0]['update'], last_update=rows[-1]['update'],
                updates=len(rows), completed_episodes=ended,
                successful_episodes=successful,
                completed_episode_success_rate=successful/ended if ended else None)


def unique_series(events):
    result = {}
    for event in events:
        if event.step in result and result[event.step] != event.value:
            raise ValueError('Conflicting resumed TensorBoard events at the same update')
        result[event.step] = event.value
    return result


def audit(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from .deployment_suite import digest, utc

    run = Path(run).resolve()
    status = json.loads((run/'rank_0_status.json').read_text())
    agent = json.loads((run/'agent_cfg.json').read_text())
    if not status.get('settings', {}).get('stones', {}).get('start_stop'):
        raise ValueError('Run does not enable the causal start/stop task')
    events = EventAccumulator(str(run), size_guidance={'scalars': 0}).Reload()
    ended = unique_series(events.Scalars(PREFIX+'ended_episodes'))
    successful = unique_series(events.Scalars(PREFIX+'successful_episodes'))
    rows = recover_counts(ended, successful, agent['num_steps_per_env'])
    expected_start = status['resumed_after_updates']
    if rows[0]['update'] != expected_start:
        raise ValueError('Count series does not begin at this run initialization')
    fractions = {}
    for name, expected in [('world_fraction', .125), ('online_fraction', .25),
                           ('command_error_max', 0.)]:
        values = unique_series(events.Scalars(PREFIX+name))
        if any(step not in values or abs(values[step]-expected) > 1e-6 for step in ended):
            raise ValueError('Start/stop sampling or velocity schedule differs from contract')
        fractions[name] = expected
    sampled = dict(ended=0, successful=0, updates=0)
    for line in (run/'rank_0_metrics.jsonl').read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue  # a running writer may be appending its final line
        step = expected_start+item['completed_updates']-1
        if step in ended:
            metrics = item['metrics']
            sampled['ended'] += metrics[PREFIX+'ended_episodes']
            sampled['successful'] += metrics[PREFIX+'successful_episodes']
            sampled['updates'] += 1
    root = Path(__file__).resolve().parents[3]
    sources = [Path(__file__), root/'vendor/rsl_rl/rsl_rl/utils/logger.py',
               Path(__file__).with_name('start_stop_task.py')]
    return dict(verified_at_utc=utc(), run=str(run), rank=0,
                scope='Completed training episodes from rank zero; all training families and assigned levels mixed',
                method='Sum of TensorBoard per-control-step count means times num_steps_per_env',
                rollout_steps=agent['num_steps_per_env'], all_available=aggregate(rows),
                recent_windows={str(n): aggregate(rows[-n:]) for n in (50, 100) if len(rows) >= n},
                contract_values=fractions, last_step_json_samples=sampled,
                per_update=rows, source_sha256={str(p): digest(p) for p in sources},
                limitations=['Only rank zero is written to this TensorBoard stream.',
                             'Training policy and terrain sampling change during the interval.',
                             'Early updates include incomplete long episodes; zero early successes are expected.',
                             'No per-family or per-level start/stop rate can be recovered from these aggregate tags.',
                             'This is not a fixed-checkpoint independent deployment evaluation.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    report = audit(args.run)
    with Path(args.output).open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ('run', 'rank', 'all_available', 'recent_windows',
                                           'last_step_json_samples')}, indent=2))


if __name__ == '__main__':
    main()
