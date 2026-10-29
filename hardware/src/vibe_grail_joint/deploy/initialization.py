"""Supported nominal-pose initialization; this is not a balance controller."""
import numpy as np


def pose_report(state, contract, *, initializing):
    """Report pose/speed differences without a near-nominal admission gate.

    Finite measurements remain mandatory. The caller must provide physical
    support: joint interpolation cannot stand up or balance a fallen robot.
    Policy activation is still a separate post-transition user command.
    """
    q, dq = np.asarray(state.q), np.asarray(state.dq)
    target = np.asarray(contract['default_joint_pos'])
    names = contract['joint_names']
    if (q.shape != (29,) or dq.shape != (29,) or target.shape != (29,)
            or len(names) != 29 or not np.isfinite([q, dq, target]).all()):
        raise ValueError('Invalid nominal-pose admission data')
    limits = np.full(29, .5)
    errors = np.abs(q-target)
    violations = []
    for i, name in enumerate(names):
        if errors[i] > limits[i]:
            violations.append(dict(joint=name, kind='pose_error', value=float(errors[i]),
                                   limit=float(limits[i]), q=float(q[i]), target=float(target[i])))
        if abs(dq[i]) > .5:
            violations.append(dict(joint=name, kind='joint_speed', value=float(abs(dq[i])), limit=.5))
    return dict(admitted=True, initializing=initializing, thresholds_diagnostic_only=True,
                max_pose_error_rad=float(errors.max()), max_abs_dq_rad_s=float(np.abs(dq).max()),
                violations=violations, q=q.tolist(), dq=dq.tolist(), target=target.tolist())


class NominalPoseTransition:
    """Quintic joint target ramp with zero endpoint velocity/acceleration.

    1.875 is the peak derivative of 10u^3 - 15u^4 + 6u^5. Duration
    therefore bounds the commanded joint speed, not the measured motor speed.
    Physical support is required throughout the transition and pose hold.
    """
    def __init__(self, start, target):
        self.start, self.target = (np.asarray(v, dtype=np.float64).copy() for v in (start, target))
        if (self.start.shape != (29,) or self.target.shape != (29,)
                or not np.isfinite([self.start, self.target]).all()):
            raise ValueError('Invalid initialization target')
        self.max_target_speed_rad_s = .25
        self.duration = max(3., 1.875*float(np.max(np.abs(self.target-self.start)))/self.max_target_speed_rad_s)

    def sample(self, elapsed):
        if not np.isfinite(elapsed):
            raise ValueError('Invalid initialization elapsed time')
        u = float(np.clip(elapsed/self.duration, 0., 1.))
        fraction = u*u*u*(10.+u*(-15.+6.*u))
        target = self.start + fraction*(self.target-self.start)
        return target.astype(np.float32), u >= 1.
