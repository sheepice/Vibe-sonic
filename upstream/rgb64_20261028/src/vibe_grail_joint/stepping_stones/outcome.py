"""Task completion is stricter than the separate forward/start/stop check."""
import math


def portable_outcome(report):
    """Require the full route AND the existing deployment start/stop protocol.

    This consumes evaluation evidence only. It is never a policy observation,
    reward or reason to relax an actuator guard.
    """
    reasons = []
    if report.get('verifies_airborne_gap_crossing'):
        if not report.get('gap_crossing_audit', {}).get('gap_crossing_qualified', False):
            reasons.append('airborne_gap_crossing_unverified')
    if not report.get('standing_start_stop_sim_passed', False):
        reasons.append('start_stop_not_passed')
    if not report.get('course_success', False):
        reasons.append('course_not_completed')
    completed = report.get('course_completion_time_s')
    duration = report.get('simulated_seconds', 0.)
    if (not isinstance(completed, (int, float)) or not math.isfinite(completed)
            or not isinstance(duration, (int, float)) or not math.isfinite(duration)
            or not 0. < completed <= duration + 1e-6):
        reasons.append('completion_time_invalid')
    if report.get('fall', True):
        reasons.append('fall_or_floor_contact')
    if report.get('guard_stop_reason') or report.get('target_guard_counts'):
        reasons.append('actuator_guard_failed')
    if not report.get('raw_rgb_encoder_used', False) or not report.get('deploy_onnx_drove_scored_world', False):
        reasons.append('portable_rgb_control_unverified')
    if report.get('family') == 'stair_gap':
        route = report.get('route_audit', {})
        if not all(route.get(key, False) for key in (
                'route_verification_required', 'departure_platform_contact',
                'landing_platform_contact', 'route_order_verified')):
            reasons.append('stair_gap_route_unverified')
    return dict(course_and_stop_passed=not reasons, task_failure_reasons=reasons)


def terminal_status(report):
    if portable_outcome(report)['course_and_stop_passed']:
        return 'PASS + STOP'
    if report.get('guard_stop_reason'):
        return report['guard_stop_reason']
    if report.get('fall', True):
        return 'FALL / FLOOR CONTACT'
    if not report.get('course_success', False):
        return 'COURSE NOT COMPLETED'
    if not report.get('standing_start_stop_sim_passed', False):
        return 'START / STOP INCOMPLETE'
    return 'TASK NOT VERIFIED'
