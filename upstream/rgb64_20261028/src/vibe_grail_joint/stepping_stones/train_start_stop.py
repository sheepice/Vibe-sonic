"""Separate two-rank entrypoint for full-state matched-start continuation.

The shared trainer is invoked with process-local task factories. Original
training entrypoints, deployment code, model size and actor inputs are unchanged.
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--revise-start-stop',action='store_true')
    selected,remaining=parser.parse_known_args()
    check=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    check.add_argument('--config',required=True)
    check.add_argument('--log-dir',required=True)
    check.add_argument('--resume',required=True)
    args,_=check.parse_known_args(remaining)
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    from . import start_stop_task,task,curriculum,train
    from .. import catalog
    from .deployment_suite import digest,write,utc
    import os
    settings=json.loads(Path(args.config).read_text())
    cache=start_stop_task.validate_start_stop_options(settings['stones'].get('start_stop'))
    out=Path(args.log_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ.get('RANK','0'))
    write(out/f'start_stop_contract_rank_{rank}.json',dict(
        created_utc=utc(),reference_cache=settings['stones']['start_stop'],
        complete_cache_clips=len(cache.clips),source_checkpoint=str(Path(args.resume).resolve()),
        actor_input_channels_unchanged=True,extra_privileged_actor_inputs=[],
        original_online_fraction=.5,matched_start_fraction_of_online=.25,
        preserves_paired_and_flat_dagger=True,model_parameter_count_unchanged=True,
        source_sha256={str(p):digest(p) for p in (Path(__file__),Path(start_stop_task.__file__))}))

    def validate(saved,current,*,allow_revision=False):
        return start_stop_task.validate_start_stop_resume(saved,current,allow_revision=allow_revision,
                                                         allow_start_stop=selected.revise_start_stop)

    original_write=catalog.write_json
    def write_contract(path,value):
        if Path(path).name=='task_contract.json':
            value=dict(value,
                online_reference='Original shared rolling SONIC streams plus causal deployment-reference snapshots for 25% of online PPO worlds; no terrain conditioning',
                start_stop_training=settings['stones']['start_stop'],
                reference_fractions_of_all_worlds=dict(online_ppo_moving=.375,online_ppo_start_stop=.125,
                    paired_grail_ppo=.2,flat_dagger_planner=.15,flat_dagger_replay=.15),
                start_stop_cache_required_on_real_robot=False,
                training_entrypoint='vibe_grail_joint.stepping_stones.train_start_stop')
        return original_write(path,value)

    with patch.object(task,'env_config',start_stop_task.env_config), \
            patch.object(task,'QualityStairsEnv',start_stop_task.StartStopStairsEnv), \
            patch.object(curriculum,'validate_resume_options',validate), \
            patch.object(catalog,'write_json',write_contract), \
            patch.object(sys,'argv',[sys.argv[0],*remaining]):
        train.main()


if __name__=='__main__':
    main()
