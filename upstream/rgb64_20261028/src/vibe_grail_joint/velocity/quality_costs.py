"""Bounded, unit-normalized training costs; no policy inputs or learned modules."""
import torch


def positive_cost(value, tolerance, scale, cap=1.):
    return ((value-tolerance).clamp_min(0)/scale).square().clamp_max(cap)


def impact_cost(impulse_ns, step_dt, force_scale_n):
    # Integrating force over ALL physics substeps avoids missing short impacts.
    return (impulse_ns/(step_dt*force_scale_n)).clamp(0, 2.).mean(-1)


def supported_slip_cost(distance, seconds, stable, tolerance=.03, scale=.15):
    speed=distance/seconds.clamp_min(1e-8)
    return (positive_cost(speed,tolerance,scale)*(stable&(seconds>0))).mean(-1)


def support_margin_cost(margin, stable, desired=.025, scale=.05):
    return (positive_cost(-margin,-desired,scale)*stable).mean(-1)


def swing_riser_cost(front_x, front_z, edges, jumps, base, swing,
                     lookahead=.18, behind=.05, clearance=.035, scale=.06):
    """Front sole must clear an upcoming upward riser; zero on flat/descent.

    Coordinates share the tile frame. Finite edge mask prevents padded infs
    from producing NaNs. Actual contact gates swing, not the flat motion phase.
    """
    finite=torch.isfinite(edges)
    dx=torch.where(finite[:,None],edges[:,None]-front_x[...,None],1e4)
    tops=base[:,None]+jumps.cumsum(-1)
    active=finite[:,None]&(jumps[:,None]>0)&(dx>=-behind)&(dx<=lookahead)
    proximity=torch.where(dx>=0,1-dx/lookahead,1+dx/behind).clamp(0,1)
    shortage=positive_cost(tops[:,None]+clearance-front_z[...,None],0.,scale)
    per_foot=(shortage*proximity*active).amax(-1)*swing
    return per_foot.mean(-1)
