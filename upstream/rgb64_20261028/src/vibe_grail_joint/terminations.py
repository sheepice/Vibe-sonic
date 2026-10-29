"""Stairs include a final hold; artificial flat windows truncate without forcing a stop."""


def completed_reference(env, command_name: str, terrain_hold_steps: int):
    command = env.command_manager.get_term(command_name)
    terrain = command.motion.terrain_ids[command._clip_ids] >= 0
    at_end = command.time_steps >= command.motion.clip_ends[command._clip_ids] - 1
    return at_end & (~terrain | (command._steps_past_end >= terrain_hold_steps))
