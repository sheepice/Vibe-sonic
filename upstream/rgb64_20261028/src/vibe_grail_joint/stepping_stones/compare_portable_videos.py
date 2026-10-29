"""Combine two matched portable trials; optionally remove only their temporary clips."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left-report',required=True);p.add_argument('--right-report',required=True)
    p.add_argument('--output',required=True);p.add_argument('--remove-temporary-singles',action='store_true')
    a=p.parse_args()
    import numpy as np
    import imageio.v2 as iio
    paths=[Path(a.left_report).resolve(),Path(a.right_report).resolve()]
    reports=[json.loads(p.read_text()) for p in paths]
    from .outcome import portable_outcome
    outcomes=[portable_outcome(r) for r in reports]
    for r,result in zip(reports,outcomes):
        if (r['video']['terminal_status']=='PASS + STOP')!=result['course_and_stop_passed']:
            raise ValueError('Recorded PASS label does not match full course-and-stop evidence')
    for key in ('family','level','seed','terrain_variant','command_speed_m_s','requested_seconds','stop_hold_seconds'):
        if reports[0][key]!=reports[1][key]:raise ValueError('Mismatched trial '+key)
    for r in reports:
        if not r['raw_rgb_encoder_used'] or r['no_physics_guards_applied'] or r['initial_state']!='stationary_default_pose':
            raise ValueError('Expected guarded portable stationary-start trials')
        if r['video']['camera_max_pose_drift']>1e-7 or not r['video']['actual_rgb_panel']:raise ValueError('Wrong recording contract')
    videos=[Path(r['video']['path']).resolve() for r in reports]
    for path,report,video in zip(paths,reports,videos):
        if video.name!='diagnostic_single.mp4' or video.parent!=path.parent or not report['video']['temporary_single_for_combination']:
            raise ValueError('Only our temporary portable diagnostic clips are accepted')
    initial_error=max(float(np.abs(np.asarray(reports[0]['video']['initial_state'][k])-reports[1]['video']['initial_state'][k]).max()) for k in reports[0]['video']['initial_state'])
    camera_error=max(float(np.abs(np.asarray(reports[0]['video'][k])-reports[1]['video'][k]).max()) for k in ('camera_position','camera_rotation'))
    if max(initial_error,camera_error)>1e-6:raise ValueError('Initial state or fixed observer differs')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    readers=[iio.get_reader(v) for v in videos];writer=None
    try:
        counts=[r.count_frames() for r in readers]
        if counts!=[r['video']['frames'] for r in reports]:raise ValueError('Frame count mismatch')
        if any(abs(r.get_meta_data()['fps']-25)>1e-6 for r in readers):raise ValueError('Expected 25 FPS')
        path=out/'comparison_fixed_rgb.mp4'
        writer=iio.get_writer(path,fps=25,codec='libx264',quality=8,macro_block_size=1,
            ffmpeg_params=['-threads','1','-pix_fmt','yuv420p'])
        last=[None,None]
        for i in range(max(counts)):
            for side in [0,1]:
                if i<counts[side]:last[side]=readers[side].get_next_data()
            if any(x.shape!=(816,960,3) for x in last):raise ValueError('Unexpected panel dimensions')
            writer.append_data(np.concatenate(last,axis=1))
        writer.close();writer=None
    finally:
        if writer:writer.close()
        for reader in readers:reader.close()
    with iio.get_reader(path) as reader:
        actual=reader.count_frames()
        if actual!=max(counts):raise RuntimeError('Combined clip failed frame verification')
        assert reader.get_data(actual-1).shape==(816,1920,3)
    summary=dict(video=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=actual,fps=25,
        only_combined_video=True,initial_state_max_abs_difference=initial_error,observer_max_abs_difference=camera_error,
        raw_rgb_portable_control=True,single_scenario_demonstration=True,hardware_validated=False,
        sides=[dict(report=str(p),bundle=r['bundle'],passed=result['course_and_stop_passed'],
                    start_stop_passed=r['standing_start_stop_sim_passed'],
                    course_success=r['course_success'],task_failure_reasons=result['task_failure_reasons'],
                    fall=r['fall'],guard_stop_reason=r.get('guard_stop_reason'),
                    course_completion_time_s=r['course_completion_time_s'],last_second=r['last_second'],
                    video_terminal_status=r['video']['terminal_status']) for p,r,result in zip(paths,reports,outcomes)])
    (out/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    if a.remove_temporary_singles:
        for p,r,video in zip(paths,reports,videos):
            video.unlink()
            r['video'].update(removed_after_combination=True,combined_video=str(path.resolve()))
            tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(p)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
