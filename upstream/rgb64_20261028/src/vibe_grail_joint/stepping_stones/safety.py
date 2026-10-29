"""Deployment target rounding correction; physical limits remain unchanged."""
import json
from pathlib import Path

import numpy as np

from ..deploy import safety as original

TARGET_PROJECTION_ROUNDING = 'inward_float32_v1'


def inward_bounds(contract):
    """Represent physical bounds in float32 without rounding outside them."""
    exact = np.asarray(contract['joint_limits'], dtype=np.float64)
    if exact.shape != (29,2) or not np.isfinite(exact).all():
        raise ValueError('Invalid physical joint bounds')
    bounded = exact.astype(np.float32)
    bounded[:, 0] = np.where(bounded[:, 0].astype(np.float64) < exact[:, 0],
                             np.nextafter(bounded[:, 0], np.float32(np.inf)), bounded[:, 0])
    bounded[:, 1] = np.where(bounded[:, 1].astype(np.float64) > exact[:, 1],
                             np.nextafter(bounded[:, 1], np.float32(-np.inf)), bounded[:, 1])
    if np.any(bounded[:,0] >= bounded[:,1]):
        raise ValueError('Invalid physical joint bounds')
    return bounded


def condition_target(target, previous, contract, limits):
    # Reuse all original thresholds and slew logic. Only the representation of
    # the clipping endpoints moves inward by at most one float32 spacing.
    clipping_contract = dict(contract, joint_limits=inward_bounds(contract))
    return original.condition_target(target, previous, clipping_contract, limits)


def validate_sim_acceptance(path, manifest, speed=None):
    report = json.loads(Path(path).read_text())
    if report.get('target_projection_rounding') != TARGET_PROJECTION_ROUNDING:
        raise ValueError('Run the portable guarded check with the current target-rounding implementation')
    return original.validate_sim_acceptance(path, manifest, speed)
