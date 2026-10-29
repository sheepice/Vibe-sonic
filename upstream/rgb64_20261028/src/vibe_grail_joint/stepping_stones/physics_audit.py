"""Fail at the first corrupt physics substep and retain a CPU replay snapshot.

This observer never modifies physics or policy inputs. Expanded MuJoCo model
fields are retained so CPU replay uses the failing world's randomized dynamics.
"""
import os
from pathlib import Path

import torch


DATA_FIELDS = ('qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'qfrc_applied',
               'xfrc_applied', 'act', 'mocap_pos', 'mocap_quat')


def install(env, directory):
    sim = env.sim
    original = sim.step
    directory = Path(directory)
    command = env.command_manager.get_term('motion')

    def audited_step():
        before = {k: getattr(sim.data, k).clone() for k in DATA_FIELDS
                  if hasattr(sim.data, k)}
        original()
        bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for k in ('qpos', 'qvel', 'qacc', 'qacc_warmstart'):
            value = getattr(sim.data, k)
            bad |= ~torch.isfinite(value).reshape(env.num_envs, -1).all(-1)
        if not bad.any():
            return
        ids = bad.nonzero().flatten()
        selected = ids[:4]
        models, errors = {}, {}
        # Capture *all* expanded parameters, including per-world mass/friction.
        for k in sorted(sim.expanded_fields):
            try:
                value = getattr(sim.model, k)
                models[k] = value[selected if len(value) == env.num_envs else [0]].cpu()
            except Exception as exc:
                errors[k] = repr(exc)
        state = dict(
            rank=int(os.environ.get('RANK', 0)), policy_step=int(env.common_step_counter),
            physics_step=int(env._sim_step_counter), ids=selected.cpu(), all_bad_ids=ids.cpu(),
            tile=command.tile[selected].cpu(), family=command.family[command.tile[selected]].cpu(),
            actions=env.action_manager.action[selected].cpu(),
            before={k: v[selected].cpu() for k, v in before.items()},
            after={k: getattr(sim.data, k)[selected].cpu() for k in before},
            qacc_after=sim.data.qacc[selected].cpu(), expanded_model=models,
            expanded_model_errors=errors,
        )
        stem = directory / f'physics_rank_{state["rank"]}_substep_{state["physics_step"]}'
        torch.save(state, stem.with_suffix('.pt'))
        import mujoco
        mujoco.mj_saveModel(sim.mj_model, str(stem.with_suffix('.mjb')), None)
        raise RuntimeError(f'First nonfinite physics state; snapshot: {stem}.pt')

    sim.step = audited_step


def main():
    """Replay one saved GPU transition using CPU MuJoCo; no GPU required."""
    import argparse, json, mujoco, numpy as np
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument('snapshot')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    path = Path(args.snapshot)
    state = torch.load(path, map_location='cpu', weights_only=False)
    result = dict(snapshot=str(path.resolve()), expanded_model_errors=state['expanded_model_errors'], cases=[])
    for slot, world in enumerate(state['ids'].tolist()):
        model = mujoco.MjModel.from_binary_path(str(path.with_suffix('.mjb')))
        unsupported = {}
        for k, value in state['expanded_model'].items():
            try:
                field = getattr(model, k)
                field[:] = value[min(slot, len(value)-1)].numpy().reshape(field.shape)
            except Exception as exc:
                unsupported[k] = repr(exc)
        data = mujoco.MjData(model)
        mujoco.mj_setConst(model, data)
        for k, value in state['before'].items():
            getattr(data, k)[:] = value[slot].numpy().reshape(getattr(data, k).shape)
        before = dict(qpos=data.qpos.copy(), qvel=data.qvel.copy())
        mujoco.mj_step(model, data)
        result['cases'].append(dict(
            world=world, family=int(state['family'][slot]), unsupported_model_fields=unsupported,
            before_max_qvel=float(np.max(np.abs(before['qvel']))),
            after_finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
            after_max_qvel=float(np.max(np.abs(data.qvel))),
            after_max_qacc=float(np.max(np.abs(data.qacc))), ncon=data.ncon,
            gpu_after_finite=bool(torch.isfinite(state['after']['qpos'][slot]).all()),
        ))
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
