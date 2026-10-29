"""Summarize a terminal fixed deployment comparison, preserving every failure."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .deployment_suite import digest, utc, validate_report


def summarize(directory, baseline, candidate):
    directory = Path(directory).resolve()
    state_path = directory/'state.json'
    state = json.loads(state_path.read_text())
    if state.get('status') != 'completed' or any(t['status'] != 'completed' for t in state['trials']):
        raise ValueError('Require a completed suite with every trial report available')
    candidates = {c['name']: c for c in state['candidates']}
    if set(candidates) != {baseline, candidate} or baseline == candidate:
        raise ValueError('Expected exactly the two specified candidates')
    for c in candidates.values():
        if digest(c['checkpoint']) != c['checkpoint_sha256'] or digest(c['config']) != c['config_sha256']:
            raise ValueError('Candidate checkpoint/config changed')
        if digest(Path(c['bundle'])/'bundle.json') != c['manifest_sha256']:
            raise ValueError('Candidate bundle manifest changed')
        for name, expected in c['bundle_files'].items():
            if digest(Path(c['bundle'])/name) != expected:
                raise ValueError('Candidate bundle weights changed')
    rows, pairs, totals = [], defaultdict(dict), {}
    for trial in state['trials']:
        path = Path(trial['output'])/'report.json'
        report = json.loads(path.read_text())
        resource = json.loads(Path(trial['output']+'.resource.json').read_text())
        if resource.get('status') != 'completed' or resource.get('exit_code') != 0:
            raise ValueError('A review process did not complete normally')
        result = validate_report(report, trial, candidates[trial['candidate']])
        if result != trial['result']:
            raise ValueError('Saved trial outcome differs from the report')
        row = {k: trial[k] for k in ('id', 'candidate', 'family', 'level', 'speed', 'seed', 'variant', 'repeat')}
        row.update(result, report=str(path), report_sha256=digest(path),
                   peak_review_mib=resource['max_review_mib'])
        rows.append(row)
        key = (trial['family'], trial['level'], trial['speed'], trial['seed'], trial['variant'], trial['repeat'])
        if trial['candidate'] in pairs[key]:
            raise ValueError('Duplicate trial in a paired comparison')
        pairs[key][trial['candidate']] = result['passed']
    for name in candidates:
        selected = [r for r in rows if r['candidate'] == name]
        families = {}
        for family in sorted({r['family'] for r in selected}):
            group = [r for r in selected if r['family'] == family]
            families[family] = dict(passed=sum(r['passed'] for r in group), total=len(group))
        repeat_groups = defaultdict(list)
        for row in selected:
            repeat_groups[(row['family'], row['level'], row['speed'])].append(row['passed'])
        totals[name] = dict(passed=sum(r['passed'] for r in selected), total=len(selected),
                            families=families, both_repeats_passed=sum(len(v)==2 and all(v) for v in repeat_groups.values()),
                            scene_speed_pairs=len(repeat_groups),
                            failure_reasons=dict(Counter(reason for r in selected for reason in r['reasons'])))
    paired = Counter()
    for result in pairs.values():
        if set(result) != set(candidates):
            raise ValueError('Missing matched candidate trial')
        a, b = result[baseline], result[candidate]
        paired['both_pass' if a and b else 'candidate_only_pass' if b else 'baseline_only_pass' if a else 'both_fail'] += 1
    return dict(verified_at_utc=utc(), suite=str(directory), state_sha256=digest(state_path),
                baseline=baseline, candidate=candidate, candidates=list(candidates.values()),
                summary=totals, paired_outcomes=dict(paired), trials=rows,
                completed_trials=len(rows), process_failures=0, retries=state['retries'],
                hardware_validated=False,
                limitation='Two repeats per scene and speed at one held-out seed/variant; not broad terrain generalization or hardware verification.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = summarize(args.suite, args.baseline, args.candidate)
    with Path(args.output).open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(summary=result['summary'], paired_outcomes=result['paired_outcomes']), indent=2))


if __name__ == '__main__':
    main()
