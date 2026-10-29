"""Instrument the unchanged evaluation rollouts with physical step-quality data."""
import argparse
import itertools
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--scene',choices=('uniform','long'),default='uniform')
    args,rest=parser.parse_known_args()
    import mjlab.envs
    import numpy as np
    import torch
    from .. import catalog
    from .step_quality_observer import add_quality_sensors,StepQualityObserver
    original=mjlab.envs.ManagerBasedRlEnv
    instances=[]

    class InstrumentedEnv(original):
        def __init__(self,cfg,**kwargs):
            add_quality_sensors(cfg)
            super().__init__(cfg,**kwargs)
            self.quality=StepQualityObserver(self)
            self.quality_trace=[]
            previous=self.metrics_manager.compute_substep
            def substep():
                previous();self.quality.observe_substep()
            self.metrics_manager.compute_substep=substep
            instances.append(self)

        def step(self,action):
            self.quality.begin_step()
            result=super().step(action)
            geometry=self.quality.geometry()
            record={**self.quality.step_data,**geometry}
            self.quality_trace.append({k:v.detach().cpu().numpy().copy() for k,v in record.items()})
            return result

    mjlab.envs.ManagerBasedRlEnv=InstrumentedEnv
    write=catalog.write_json

    def augmented_write(path,value):
        write(path,value)
        if Path(path).name!='report.json':return
        env=instances[-1];trace=env.quality_trace
        arrays={k:np.stack([r[k] for r in trace]) for k in trace[0]}
        arrays['time_s']=np.arange(1,len(trace)+1)*env.step_dt
        np.savez_compressed(Path(path).parent/'step_quality_trace.npz',**arrays)
        summaries=[]
        for i,case in enumerate(value['cases']):
            count=min(len(trace),case['step'])
            def a(k):return arrays[k][:count,i]
            support=a('stable_support')
            margins=a('sole_edge_margin_m')[support]
            wall=a('foot_riser_force_n')>15.
            shin=a('shin_force_n')>15.
            wall_starts=wall&~np.concatenate((np.zeros((1,2),bool),wall[:-1]))
            shin_starts=shin&~np.concatenate((np.zeros((1,2),bool),shin[:-1]))
            support_seconds=float(a('support_seconds').sum())
            touchdown=a('touchdowns')>0
            incoming=a('touchdown_vz_m_s')[touchdown]
            summary=dict(case=case['case'],success=case['success'],time_s=case['time_s'],
                foot_riser_strikes=int(wall_starts.sum()),shin_strikes=int(shin_starts.sum()),
                foot_riser_max_force_n=float(a('foot_riser_force_n').max()),
                shin_max_force_n=float(a('shin_force_n').max()),
                foot_riser_total_impulse_ns=float(a('foot_riser_impulse_ns').sum()),
                shin_total_impulse_ns=float(a('shin_impulse_ns').sum()),
                support_slip_distance_m=float(a('slip_distance_m').sum()),
                support_seconds=support_seconds,
                support_mean_slip_speed_m_s=float(a('slip_distance_m').sum())/max(support_seconds,1e-8),
                supported_foot_samples=int(support.sum()),
                edge_margin_p05_m=float(np.quantile(margins,.05)) if len(margins) else None,
                edge_margin_median_m=float(np.median(margins)) if len(margins) else None,
                support_overhang_fraction=float((margins<0).mean()) if len(margins) else None,
                support_margin_under_2cm_fraction=float((margins<.02).mean()) if len(margins) else None,
                touchdowns=int(touchdown.sum()),
                touchdown_vz_p05_m_s=float(np.quantile(incoming,.05)) if len(incoming) else None,
                collision_events=[dict(time_s=float(arrays['time_s'][t]),foot=int(f),
                    root_x_m=float(arrays['root_local'][t,i,0]),
                    force_n=float(arrays['foot_riser_force_n'][t,i,f]),
                    body='foot') for t,f in np.argwhere(wall_starts)]
                    +[dict(time_s=float(arrays['time_s'][t]),foot=int(f),
                    root_x_m=float(arrays['root_local'][t,i,0]),
                    force_n=float(arrays['shin_force_n'][t,i,f]),
                    body='shin') for t,f in np.argwhere(shin_starts)])
            summaries.append(summary)
        write(Path(path).parent/'step_quality_summary.json',dict(
            checkpoint=value['checkpoint'],checkpoint_iteration=value['checkpoint_iteration'],
            measurement='Contact normals, positions and forces sampled every physics substep; traces reduced at 50Hz.',
            strike_threshold_n=15.,stable_support_steps=3,support_force_threshold_n=20.,
            sole_geometry='Conservative projected full-foot rectangle, 20.6 x 7.4 cm; margin is not contact area.',
            cases=summaries))

    catalog.write_json=augmented_write
    if args.scene=='uniform':
        from . import uniform_probe20
        def cases(settings):
            rows=[]
            for kind,speed,phase in itertools.product(range(4),(.25,.4,.7),(0,5,10,15)):
                stream=settings['planner_speeds'].index(speed)
                rows.append(dict(id=len(rows),tile=kind,stream=stream,speed=speed,phase=phase,
                                 spawn_xy=[-.15+phase*.02,-.15+phase*.02]))
            return rows
        uniform_probe20.probe_cases=cases
        from .evaluate_uniform20 import main as evaluate
        rest+=['--uniform-probe-20cm']
    else:
        from .evaluate_long_walk import main as evaluate
        rest+=['--long-walk-probe','--record-case','0','--record-speed','0.4']
    sys.argv=[sys.argv[0],*rest]
    evaluate()


if __name__=='__main__':main()
