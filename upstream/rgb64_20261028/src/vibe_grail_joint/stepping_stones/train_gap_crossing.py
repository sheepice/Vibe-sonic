"""Isolated RGB continuation with an explicit change to gap success semantics."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


def validate_migration(saved, current, *, allow_gap=False):
    from .start_stop_task import base_validate_resume_options
    from .gap_crossing import options
    old = saved.get('stones', {}).get('gap_crossing') if saved else None
    new = current.get('stones', {}).get('gap_crossing')
    options(new)
    if old == new:
        if allow_gap:
            raise ValueError('Already migrated; resume without --revise-gap-crossing')
        return base_validate_resume_options(saved, current)
    if not allow_gap or old is not None or saved is None:
        raise ValueError('Explicit --revise-gap-crossing required for first migration')
    stripped = copy.deepcopy(current)
    stripped['stones'].pop('gap_crossing')
    if stripped != saved:
        raise ValueError('Gap migration cannot also change other task or actor options')
    return dict(gap_crossing_changed=True, actor_inputs_changed=False,
                terrain_changed=False, rewards_and_success_changed=True,
                sparse_curriculum_history='reset; previous bridge-compatible successes are invalid')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--revise-gap-crossing', action='store_true')
    selected, remaining = parser.parse_known_args()
    check = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    check.add_argument('--config', required=True)
    check.add_argument('--log-dir', required=True)
    args, _ = check.parse_known_args(remaining)
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    from . import start_stop_task, train_start_stop, gap_crossing_task
    from .gap_crossing import options
    from .deployment_suite import write, utc, digest
    settings = json.loads(Path(args.config).read_text())
    contract = options(settings['stones']['gap_crossing'])
    old_config, old_env = start_stop_task.env_config, start_stop_task.StartStopStairsEnv

    def config(bank, settings, n):
        return gap_crossing_task.configure(old_config(bank, settings, n), settings)

    def environment(cfg, *, device, settings):
        env = old_env(cfg, device=device, settings=settings)
        gap_crossing_task.install(env, settings)
        original_reset = env.reset
        migrated = False
        def reset(*args, **kwargs):
            nonlocal migrated
            if selected.revise_gap_crossing and not migrated and env.common_step_counter > 0:
                c = env.command_manager.get_term('motion')
                c.frontier[1:] = 0
                c.course_counts[1:] = 0; c.course_total[1:] = 0
                c.last_promoted[1:] = env.policy_update_count
                c.course_recent_counts[:, 1:] = 0
                migrated = True
            return original_reset(*args, **kwargs)
        env.reset = reset
        return env

    def validate(saved, current, *, allow_revision=False, allow_start_stop=False):
        if allow_revision or allow_start_stop:
            raise ValueError('Do not combine gap migration with another task migration')
        return validate_migration(saved, current, allow_gap=selected.revise_gap_crossing)

    out = Path(args.log_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    write(out/f'gap_crossing_contract_rank_{os.environ.get("RANK", "0")}.json', dict(
        utc=utc(), settings=contract, migration=selected.revise_gap_crossing,
        actor_inputs_added=[], ordinary_stairs_rewards_changed=False,
        sparse_curriculum_reset_on_migration=True,
        source_sha256={str(p):digest(p) for p in (Path(__file__), Path(gap_crossing_task.__file__))}))
    with patch.object(start_stop_task, 'env_config', config), \
            patch.object(start_stop_task, 'StartStopStairsEnv', environment), \
            patch.object(start_stop_task, 'validate_start_stop_resume', validate), \
            patch.object(sys, 'argv', [sys.argv[0], *remaining]):
        train_start_stop.main()


if __name__ == '__main__':
    main()
