"""Optional recent-window curriculum. Privileged training sampler state only."""
import copy
import math


RECENT_STATE = ('course_recent_counts', 'course_recent_cursor', 'course_recent_windows')


def recent_options(value):
    if value is None:
        return None
    expected = {'kind', 'windows', 'minimum_windows', 'minimum_episodes',
                'success_rate', 'minimum_update_interval'}
    if set(value) != expected or value['kind'] != 'recent_windows_v1':
        raise ValueError('Invalid recent curriculum configuration')
    out = copy.deepcopy(value)
    for key in ('windows', 'minimum_windows', 'minimum_episodes', 'minimum_update_interval'):
        if type(out[key]) is not int or out[key] <= 0:
            raise ValueError(f'Invalid curriculum integer {key}')
    if not 2 <= out['minimum_windows'] <= out['windows'] <= 8:
        raise ValueError('Require 2..8 curriculum windows')
    if out['minimum_episodes'] < 80 or out['minimum_update_interval'] < 150:
        raise ValueError('Do not lower the original evidence/spacing requirements')
    if not math.isfinite(out['success_rate']) or not .72 <= out['success_rate'] <= 1.:
        raise ValueError('Do not lower the original promotion success threshold')
    return out


def validate_resume_options(saved, current, *, allow_revision=False):
    """Only an explicitly requested sampler revision may differ on full restore."""
    previous = recent_options(saved.get('stones', {}).get('recent_curriculum')) if saved else None
    new = recent_options(current.get('stones', {}).get('recent_curriculum'))
    changed = previous != new
    if saved != current:
        old_copy, new_copy = copy.deepcopy(saved), copy.deepcopy(current)
        if not allow_revision or not changed or old_copy is None:
            raise ValueError('Resume requires matching task options, or explicit --revise-curriculum')
        old_copy['stones'].pop('recent_curriculum', None)
        new_copy['stones'].pop('recent_curriculum', None)
        if old_copy != new_copy:
            raise ValueError('Curriculum revision cannot change terrain, reward, actor or other task options')
        # Ring-buffer shape/interpretation changes need an explicit migration;
        # the supported revision is enabling it on a legacy cumulative run.
        if previous is not None or new is None:
            raise ValueError('Only cumulative -> recent-window revision is supported')
    return dict(previous=previous, current=new, curriculum_changed=changed,
                actor_inputs_changed=False, terrain_or_rewards_changed=False,
                new_window_history='empty; collect fresh evidence' if changed else 'restore saved state')


def initialize_recent(command, value):
    import torch
    command.recent_curriculum = recent_options(value)
    if command.recent_curriculum is None:
        return
    command.course_recent_counts = torch.zeros(
        command.recent_curriculum['windows'], 4, 11, 2, device=command.device)
    command.course_recent_cursor = torch.zeros((), dtype=torch.long, device=command.device)
    command.course_recent_windows = torch.zeros((), dtype=torch.long, device=command.device)


def update_recent(command, counts, update, families):
    """counts have already been all-reduced and added to the lifetime totals."""
    import torch
    options = command.recent_curriculum
    if (counts.shape != (4, 11, 2) or not torch.isfinite(counts).all()
            or (counts < 0).any() or (counts[..., 1] > counts[..., 0]).any()):
        raise ValueError('Invalid completed-episode counts')
    cursor = int(command.course_recent_cursor)
    command.course_recent_counts[cursor].copy_(counts)
    command.course_recent_cursor.fill_((cursor+1) % options['windows'])
    command.course_recent_windows.add_(1).clamp_(max=options['windows'])
    recent = command.course_recent_counts.sum(0)
    nonempty = (command.course_recent_counts[..., 0] > 0).sum(0)
    result = {}
    for family, name in enumerate(families):
        level = int(command.frontier[family])
        episodes, successes = recent[family, level]
        rate = float(successes/episodes.clamp_min(1))
        evidence_windows = int(nonempty[family, level])
        promoted = bool(family and level < command.cfg.stone_levels-1
                        and episodes >= options['minimum_episodes']
                        and evidence_windows >= options['minimum_windows']
                        and rate >= options['success_rate']
                        and update-int(command.last_promoted[family]) >= options['minimum_update_interval'])
        if promoted:
            command.frontier[family] += 1
            command.last_promoted[family] = update
        prefix = f'Course/{name}/'
        result.update({prefix+'decision_level': level,
                       prefix+'decision_success_rate': rate,
                       prefix+'decision_episodes': float(episodes),
                       prefix+'decision_windows': evidence_windows,
                       prefix+'promoted': float(promoted)})
        # Reported current-frontier statistics must describe the NEW frontier
        # after promotion, rather than silently retaining the previous level.
        current = int(command.frontier[family])
        total_n, total_s = command.course_total[family, current]
        recent_n, recent_s = recent[family, current]
        result.update({prefix+'frontier': current,
                       prefix+'frontier_success_rate': float(total_s/total_n.clamp_min(1)),
                       prefix+'frontier_episodes': float(total_n),
                       prefix+'recent_success_rate': float(recent_s/recent_n.clamp_min(1)),
                       prefix+'recent_episodes': float(recent_n)})
    return result
