"""Read-only simulator actuation metrics; never fed to a control policy."""
import torch


class ActuationAudit:
    def __init__(self, env, contract):
        self.robot = env.scene['robot']
        self.names = contract['joint_names']
        if list(self.robot.joint_names) != self.names:
            raise ValueError('Actuation diagnostic joint order mismatch')
        self.device = env.device
        self.bounds = torch.tensor(contract['joint_limits'], device=self.device)
        self.effort = torch.tensor(contract['effort_limit'], device=self.device)
        if not torch.isfinite(self.effort).all() or not (self.effort > 0).all():
            raise ValueError('Actuation audit requires finite modeled effort limits')
        self.count = 0
        self.peak_torque = torch.zeros(29, device=self.device)
        self.peak_velocity = torch.zeros_like(self.peak_torque)
        self.peak_position_violation = torch.zeros_like(self.peak_torque)
        self.near_limit_count = torch.zeros_like(self.peak_torque)
        self.effort_excess_count = torch.zeros_like(self.peak_torque)
        self.nonfinite_seen = torch.zeros((), dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def observe_substep(self):
        q = self.robot.data.joint_pos[0]
        dq = self.robot.data.joint_vel[0]
        torque = self.robot.data.qfrc_actuator[0]
        if q.shape != (29,) or torque.shape != (29,):
            raise ValueError('Expected 29 joint-space quantities')
        self.nonfinite_seen |= ~(torch.isfinite(q).all() & torch.isfinite(dq).all()
                                 & torch.isfinite(torque).all())
        self.peak_torque = torch.maximum(self.peak_torque, torque.abs())
        self.peak_velocity = torch.maximum(self.peak_velocity, dq.abs())
        violation = torch.maximum(self.bounds[:, 0]-q, q-self.bounds[:, 1]).clamp_min(0)
        self.peak_position_violation = torch.maximum(self.peak_position_violation, violation)
        self.near_limit_count += torque.abs() >= .99*self.effort
        self.effort_excess_count += torque.abs() > self.effort+1e-4
        self.count += 1

    def summary(self):
        n = max(1, self.count)
        rows = []
        arrays = [x.cpu().tolist() for x in (
            self.effort, self.peak_torque, self.peak_velocity,
            self.peak_position_violation, self.near_limit_count/n,
            self.effort_excess_count/n)]
        for i, name in enumerate(self.names):
            rows.append(dict(joint=name, modeled_effort_limit_nm=arrays[0][i],
                             peak_joint_actuator_torque_nm=arrays[1][i],
                             peak_joint_velocity_rad_s=arrays[2][i],
                             peak_measured_range_violation_rad=arrays[3][i],
                             near_modeled_effort_limit_fraction=arrays[4][i],
                             above_modeled_effort_limit_fraction=arrays[5][i]))
        return dict(physics_substeps=self.count, scored_world=0,
                    uses_joint_space_qfrc_actuator=True,
                    policy_input=False, modifies_control=False,
                    nonfinite_seen=bool(self.nonfinite_seen),
                    peak_measured_range_violation_rad=float(self.peak_position_violation.max()),
                    maximum_torque_to_modeled_effort_ratio=float((self.peak_torque/self.effort).max()),
                    joints=rows, hardware_validated=False,
                    interpretation='Simulator joint actuator forces and modeled ranges only; not firmware torque verification or a hardware admission test.')
