"""Judge full stair traversal from physical state and actual foot contacts."""
import numpy as np


def evaluate(trace, tile, model):
    if not trace:
        return {"passed":False,"reason":"empty trajectory"}
    stair_names={model.geom(i).name for i in range(model.ngeom) if model.geom(i).name.startswith("stair_")}
    touched=set()
    for row in trace:
        for a,b in row["stair_contacts"]:
            if a.startswith("stair_") and "_foot" in b: touched.add(a)
            if b.startswith("stair_") and "_foot" in a: touched.add(b)
    reached=next((row["t"] for row in trace if row["reached"]),None)
    stop=[row for row in trace if row["t"]>=trace[-1]["t"]-1]
    drift=float(np.linalg.norm(np.array(stop[-1]["position"][:2])-stop[0]["position"][:2]))
    checks=dict(
        no_fall=all(not row["fall"] for row in trace),
        stayed_on_stair_width=all(abs(row["position"][1])<1.8 for row in trace),
        feet_contacted_every_stair=touched==stair_names,
        both_feet_past_goal=all(foot[0]>tile["goal"] for foot in trace[-1]["feet"]),
        four_seconds_after_goal=reached is not None and trace[-1]["t"]>=reached+4-1e-6,
        command_stopped=trace[-1]["speed_command"]==0,
        final_second_drift_under_5cm=drift<.05,
        stable_height=trace[-1]["clearance"]>.5,
    )
    return dict(passed=all(checks.values()),checks=checks,contacted_stairs=sorted(touched),
                expected_stairs=sorted(stair_names),last_second_drift_m=drift,
                max_lateral_offset_m=max(abs(row["position"][1]) for row in trace),
                min_upright=min(row["upright"] for row in trace),
                max_root_height_m=max(row["position"][2] for row in trace))
