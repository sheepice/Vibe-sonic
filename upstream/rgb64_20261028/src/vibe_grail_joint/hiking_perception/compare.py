"""Compare all scheduled evaluation cases and pre-collision action responses."""
import argparse
import json
from pathlib import Path
import numpy as np
from .contracts import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(); folders=[Path(args.reference),Path(args.candidate)]
    reports=[json.loads((d/'report.json').read_text()) for d in folders]
    if [x['case'] for x in reports[0]['cases']] != [x['case'] for x in reports[1]['cases']]: raise ValueError('Mismatched cases')
    if reports[0]['args']['seed'] != reports[1]['args']['seed']: raise ValueError('Mismatched seeds')
    traces=[np.load(d/'trace.npz') for d in folders]
    initial_equal=(all(np.allclose(reports[0]['initial_state'][k],reports[1]['initial_state'][k],atol=1e-6,rtol=0.) for k in ('root','joints'))
                   if all('initial_state' in r for r in reports) else None)
    rows=[]
    for i,(a,b) in enumerate(zip(reports[0]['cases'],reports[1]['cases'])):
        stop=min(a['steps'],b['steps']); collision=[]
        for tr in traces:
            hits=(tr['foot_riser_force_n'][:stop,i]>15).any(-1)|(tr['shin_force_n'][:stop,i]>15).any(-1)
            collision.append(int(np.flatnonzero(hits)[0]) if hits.any() else stop)
        pre=min(collision); actions=[tr['action'][:pre,i] for tr in traces]
        rows.append(dict(case=a['case'],reference_success=a['success'],candidate_success=b['success'],
            foot_impulse_delta_ns=b['foot_impulse_ns']-a['foot_impulse_ns'],shin_impulse_delta_ns=b['shin_impulse_ns']-a['shin_impulse_ns'],
            slip_speed_delta_m_s=b['support_slip_speed_m_s']-a['support_slip_speed_m_s'],speed_error_delta_m_s=b['speed_error_m_s']-a['speed_error_m_s'],
            pre_collision_steps=pre,pre_collision_action_rmse=float(np.sqrt(((actions[0]-actions[1])**2).mean())) if pre else None))
    write_json(args.output,dict(reference=str(folders[0]),candidate=str(folders[1]),initial_robot_states_equal=initial_equal,cases=rows,
        reference_completed=reports[0]['completed'],candidate_completed=reports[1]['completed'],
        note='Pre-collision action differences are evidence of a response, not proof of beneficial visual reasoning; consider direction, geometry, posture, collisions and domain-shift controls.'))


if __name__=='__main__': main()
