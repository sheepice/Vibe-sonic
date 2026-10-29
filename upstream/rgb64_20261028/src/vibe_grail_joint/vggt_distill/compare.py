"""Matched-case comparison; failed trials remain in every contact-quality average."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare_reports(left, right):
    routes = [r.get('route', 'up' if r.get('ascent_only') else None) for r in (left, right)]
    if routes[0] not in ('up', 'down', 'up_down') or routes[0] != routes[1]:
        raise ValueError('Reports must use the same supported stair route')
    modes = [r.get('completion_mode', 'strict') for r in (left, right)]
    if modes[0] not in ('strict', 'walking') or modes[0] != modes[1]:
        raise ValueError('Completion criteria differ between reports')
    if [r['case'] for r in left['cases']] != [r['case'] for r in right['cases']] or left['tiles'] != right['tiles']:
        raise ValueError('Cases or physical terrain differ between reports')
    for key in ('root', 'joints'):
        if not np.allclose(left['initial_state'][key], right['initial_state'][key], atol=1e-6, rtol=0):
            raise ValueError('Initial robot states differ')
    results = []
    for height in sorted({r['case']['nominal_height_m'] for r in left['cases']}):
        row = dict(height_m=height, out_of_training_height=height > .25)
        for name, report in (('left', left), ('right', right)):
            cases = [r for r in report['cases'] if r['case']['nominal_height_m'] == height]
            row[name] = dict(count=len(cases), success_rate=float(np.mean([r['success'] for r in cases])),
                collision_free_success_rate=float(np.mean([r['success'] and r['collision_free'] for r in cases])),
                mean_foot_impulse_ns=float(np.mean([r['foot_impulse_ns'] for r in cases])),
                mean_shin_impulse_ns=float(np.mean([r['shin_impulse_ns'] for r in cases])),
                mean_speed_error_m_s=float(np.mean([r['speed_error_m_s'] for r in cases])))
        results.append(row)
    return dict(left_checkpoint=left['checkpoint'], right_checkpoint=right['checkpoint'],
                left_iteration=left['checkpoint_iteration'], right_iteration=right['checkpoint_iteration'],
                route=routes[0], completion_mode=modes[0],
                matched_initial_states=True, matched_cases=True, by_height=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', required=True); p.add_argument('--right', required=True)
    p.add_argument('--output', required=True); p.add_argument('--allow-different-iterations', action='store_true')
    args = p.parse_args()
    left, right = [json.loads(Path(path).read_text()) for path in (args.left, args.right)]
    if not args.allow_different_iterations and left['checkpoint_iteration'] != right['checkpoint_iteration']:
        p.error('Choose equal-update checkpoints, or explicitly allow unequal training budgets')
    result = compare_reports(left, right)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
