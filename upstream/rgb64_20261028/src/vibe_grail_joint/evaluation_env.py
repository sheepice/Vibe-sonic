"""Capture terminal measurements before mjlab's automatic reset overwrites state."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv


class EvaluationEnv(ManagerBasedRlEnv):
    record_terminals = False

    def _reset_idx(self, env_ids=None):
        if self.record_terminals and env_ids is not None:
            command = self.command_manager.get_term("motion")
            robot = self.scene["robot"]
            for index in env_ids.tolist():
                clip = command._clip_ids[index]
                first, last = command.motion.clip_offsets[clip], command.motion.clip_ends[clip] - 1
                goal = command.motion.body_pos_w[last, 0] + self.scene.env_origins[index]
                root = robot.data.root_link_pos_w[index]
                current_ref = (
                    command.motion.body_pos_w[command.time_steps[index], 0] + self.scene.env_origins[index]
                )
                reasons = [
                    name
                    for name in self.termination_manager.active_terms
                    if bool(self.termination_manager.get_term(name)[index])
                ]
                failure = bool(self.termination_manager.terminated[index])
                goal_error = float((root - goal).norm())
                ref_gain = float(
                    command.motion.body_pos_w[last, 0, 2] - command.motion.body_pos_w[first, 0, 2]
                )
                actual_gain = float(
                    root[2] - command.motion.body_pos_w[first, 0, 2] - self.scene.env_origins[index, 2]
                )
                upright = float(robot.data.projected_gravity_b[index, 2]) < -0.7
                complete = "exceeded_motion" in reasons and not failure and goal_error < 0.3 and upright
                self.terminal_records[index] = {
                    "termination_reasons": reasons,
                    "reference_height_gain_m": ref_gain,
                    "robot_height_gain_m": actual_gain,
                    "final_root_error_m": goal_error,
                    "terminal_tracking_error_m": float((root - current_ref).norm()),
                    "reference_is_ascent": ref_gain > 0.2,
                    "traversal_completed": complete,
                    "ascent_success": complete and ref_gain > 0.2 and actual_gain >= ref_gain - 0.15,
                }
        super()._reset_idx(env_ids)
