"""Bounded-memory GPU aggregation for :mod:`posture_diagnostics` samples.

Typical evaluator integration (sample before new terminal detection)::

    recorder = PostureDiagnostics(command)
    summary = PostureSummary(num_envs, device, keep_trace=num_envs == 2)
    # active_mask excludes environments that terminated on earlier steps.
    summary.update(recorder.sample(elapsed_dt_s=.02), step / 50, active_mask)
    report = summary.finalize()
    trace = summary.trace  # [] unless keep_trace=True; 25 Hz for 50 Hz updates.

Statistics include the terminal sample when callers mark it active. Subsequent
samples never alter that environment's endpoint or aggregates. Statistics are
sample-weighted, not averages of episode averages. Invalid scalar samples are
excluded separately for each metric, and their counts remain visible.

No CPU transfers occur during update. Trace storage is permitted only for two
environments and is decimated by two; its CPU conversion is batched on access.
Common-window comparisons use every control-step sample while BOTH worlds are
active, independently of trace decimation. This prevents comparing a fallen
SONIC's post-terminal motion with a ViBe's remaining successful trajectory.
"""

from __future__ import annotations

import math

import torch

from .posture_diagnostics import TERRAIN_STAGE_NAMES


_ENDPOINT_KEYS = (
    "elapsed_s", "forward_displacement_m", "lateral_displacement_m",
    "commanded_distance_m", "distance_error_m", "horizontal_path_length_m",
)
_CATEGORICAL_KEYS = {"terrain_stage", "tile", "online"}
_SCOPE_NAMES = ("all", "flat_or_landing", "up", "down", "common_window")


def _number(value):
    """Strict-JSON scalar: nonfinite values have an explicit null representation."""
    value = float(value)
    return value if math.isfinite(value) else None


class PostureSummary:
    """Aggregate fixed-environment diagnostics; memory is O(envs × metrics).

    ``update(metrics, time_s, active_mask)`` accepts PostureDiagnostics output.
    It does not own episode resets: use one collector per evaluation batch.
    ``finalize()`` may be called repeatedly and does not consume stored state.
    ``trace`` is a JSON-compatible list, separate from the summary to avoid
    duplicating a large trace inside report.json.
    """

    def __init__(self, num_envs, device, keep_trace=False):
        self.num_envs, self.device = int(num_envs), torch.device(device)
        self.keep_trace = bool(keep_trace)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.keep_trace and self.num_envs != 2:
            raise ValueError("keep_trace is restricted to the two-world SONIC/ViBe comparison")
        self.metric_keys = None
        self.endpoint_keys = None
        self._scope_count = 5 if self.keep_trace else 4
        self._updates = 0
        self._trace_rows = []
        self._trace_cache = None
        self._last_call_time = None

    def _initialize(self, metrics):
        self.metric_keys = sorted(key for key in metrics
                                  if key not in _ENDPOINT_KEYS and key not in _CATEGORICAL_KEYS)
        self.endpoint_keys = [key for key in _ENDPOINT_KEYS if key in metrics]
        if not self.metric_keys:
            raise ValueError("No numeric diagnostic metrics were supplied")
        shape = (self._scope_count, self.num_envs, len(self.metric_keys))
        make = lambda: torch.zeros(shape, dtype=torch.float64, device=self.device)
        self._sum, self._absolute_sum, self._square_sum = make(), make(), make()
        self._finite_count = torch.zeros(shape, dtype=torch.long, device=self.device)
        self._maximum = torch.full(shape, -float("inf"), dtype=torch.float64, device=self.device)
        self._absolute_maximum = torch.zeros(shape, dtype=torch.float64, device=self.device)
        scope_shape = shape[:2]
        self._sample_count = torch.zeros(scope_shape, dtype=torch.long, device=self.device)
        self._first_time = torch.full(scope_shape, float("inf"), dtype=torch.float64, device=self.device)
        self._last_time = torch.full(scope_shape, -float("inf"), dtype=torch.float64, device=self.device)
        self._endpoint = torch.full((self.num_envs, len(self.endpoint_keys)), float("nan"),
                                    dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def update(self, metrics, time_s, active_mask):
        """Accumulate one synchronous sample; active mask is one bool per env."""
        time_s = float(time_s)
        if not math.isfinite(time_s) or (self._last_call_time is not None and time_s < self._last_call_time):
            raise ValueError("time_s must be finite and nondecreasing")
        active = torch.as_tensor(active_mask, device=self.device, dtype=torch.bool)
        if active.shape != (self.num_envs,):
            raise ValueError("active_mask must have shape (num_envs,)")
        # Preserve the raw actual/reference angles and also summarize their
        # signed difference directly; subtraction of two means cannot give RMSE.
        values = dict(metrics)
        for body in ("pelvis", "torso"):
            for angle in ("pitch", "roll"):
                key, ref_key = f"{body}_{angle}_deg", f"reference_{body}_{angle}_deg"
                if key in values and ref_key in values:
                    values[f"{body}_{angle}_reference_signed_error_deg"] = (
                        (values[key] - values[ref_key] + 180) % 360 - 180
                    )
        if self.metric_keys is None:
            self._initialize(values)
        numeric = torch.stack([values[key].detach() for key in self.metric_keys], dim=-1).to(
            device=self.device, dtype=torch.float64)
        if numeric.shape != (self.num_envs, len(self.metric_keys)):
            raise ValueError("Each diagnostic must be a scalar per environment")
        stage = torch.as_tensor(values["terrain_stage"], device=self.device, dtype=torch.long)
        if stage.shape != (self.num_envs,):
            raise ValueError("terrain_stage must have shape (num_envs,)")
        scope_masks = [active, active & (stage == 0), active & (stage == 1), active & (stage == 2)]
        if self.keep_trace:
            scope_masks.append(active & active.all())
        scope = torch.stack(scope_masks, dim=0)
        finite = scope[..., None] & torch.isfinite(numeric)[None, ...]
        safe = torch.where(finite, numeric[None, ...], 0.0)
        self._sample_count += scope.long()
        self._finite_count += finite.long()
        self._sum += safe
        self._absolute_sum += safe.abs()
        self._square_sum += safe.square()
        self._maximum = torch.maximum(self._maximum, torch.where(finite, numeric[None, ...], -float("inf")))
        self._absolute_maximum = torch.maximum(self._absolute_maximum, safe.abs())
        self._first_time = torch.where(scope, self._first_time.clamp(max=time_s), self._first_time)
        self._last_time = torch.where(scope, time_s, self._last_time)
        if self.endpoint_keys:
            endpoint = torch.stack([values[key].detach() for key in self.endpoint_keys], -1).to(
                device=self.device, dtype=torch.float64)
            self._endpoint = torch.where(active[:, None], endpoint, self._endpoint)
        if self.keep_trace and self._updates % 2 == 0:
            # Copy only the two-world decimated trace. Inactive values remain in
            # the trace for transparency; its mask must be applied by consumers.
            metadata = torch.stack((active, stage), -1).to(torch.float64)
            self._trace_rows.append((time_s, torch.cat((numeric, self._endpoint, metadata), -1).clone()))
            self._trace_cache = None
        self._updates += 1
        self._last_call_time = time_s

    def _stats_cpu(self, *, pooled=False):
        if pooled:
            count = self._finite_count.sum(1)
            total, absolute, squared = self._sum.sum(1), self._absolute_sum.sum(1), self._square_sum.sum(1)
            maximum, abs_max = self._maximum.max(1).values, self._absolute_maximum.max(1).values
        else:
            count = self._finite_count
            total, absolute, squared = self._sum, self._absolute_sum, self._square_sum
            maximum, abs_max = self._maximum, self._absolute_maximum
        denominator = count.clamp(min=1)
        tensor = torch.stack((count, total / denominator, absolute / denominator,
                              (squared / denominator).sqrt(), maximum, abs_max), -1)
        return tensor.cpu().tolist()

    def _format_metrics(self, array, n_samples):
        names = ("mean", "abs_mean", "rms", "max", "abs_max")
        return {
            key: {"count": int(row[0]), "invalid_count": int(n_samples - row[0]),
                  **{name: _number(value) if row[0] else None for name, value in zip(names, row[1:])}}
            for key, row in zip(self.metric_keys, array)
        }

    def finalize(self):
        """Return JSON-compatible per-env, per-stage, and pooled statistics."""
        if self.metric_keys is None:
            return {"schema_version": 1, "updates": 0, "per_env": [], "by_stage": {}, "common_window": None}
        stats, pooled = self._stats_cpu(), self._stats_cpu(pooled=True)
        count = self._sample_count.cpu().tolist()
        first, last = self._first_time.cpu().tolist(), self._last_time.cpu().tolist()
        endpoint = self._endpoint.cpu().tolist()
        per_env = []
        for env_id in range(self.num_envs):
            stages = {}
            for scope, name in enumerate(_SCOPE_NAMES[:4]):
                n = count[scope][env_id]
                stages[name] = {
                    "sample_count": n,
                    "first_time_s": _number(first[scope][env_id]),
                    "last_time_s": _number(last[scope][env_id]),
                    "metrics": self._format_metrics(stats[scope][env_id], n),
                }
            per_env.append({
                "env_index": env_id,
                "sample_count": count[0][env_id],
                "stages": stages,
                "endpoint": {key: _number(value) for key, value in zip(self.endpoint_keys, endpoint[env_id])},
            })
        by_stage = {}
        for scope, name in enumerate(_SCOPE_NAMES[:4]):
            n = sum(count[scope])
            by_stage[name] = {"sample_count": n, "metrics": self._format_metrics(pooled[scope], n)}
        common = None
        if self.keep_trace:
            common = {
                "definition": "All control-step samples while both environments are active; no post-terminal samples.",
                "sample_count_per_env": count[4][0],
                "first_time_s": _number(first[4][0]),
                "last_time_s": _number(last[4][0]),
                "per_env": [{"env_index": i, "metrics": self._format_metrics(stats[4][i], count[4][i])}
                            for i in range(2)],
            }
        return {
            "schema_version": 1,
            "updates": self._updates,
            "metric_names": self.metric_keys,
            "aggregation": "Per-control-step sample-weighted; each metric excludes nonfinite samples with invalid_count reported.",
            "angle_units": "degrees; joint velocity errors are radians/second",
            "max_definition": "max is signed maximum; abs_max is maximum absolute value",
            "stage_definition": "Root-X geometric segment, not contact state; stages 0/1/2 only; all also includes unknown/outside tile.",
            "terrain_stage_names": TERRAIN_STAGE_NAMES,
            "per_env": per_env,
            "by_stage": by_stage,
            "common_window": common,
            "trace_sample_stride": 2 if self.keep_trace else None,
            "trace_samples": len(self._trace_rows),
        }

    @property
    def trace(self):
        """Decimated JSON-compatible rows; apply active_mask to numeric metrics."""
        if not self._trace_rows:
            return []
        if self._trace_cache is None:
            payload = torch.stack([row[1] for row in self._trace_rows]).cpu().tolist()
            keys = self.metric_keys + self.endpoint_keys
            rows = []
            for (time_s, _), block in zip(self._trace_rows, payload):
                row = {"time_s": time_s,
                       "active_mask": [bool(values[-2]) for values in block],
                       "terrain_stage": [int(values[-1]) for values in block]}
                for i, key in enumerate(keys):
                    row[key] = [_number(values[i]) for values in block]
                rows.append(row)
            self._trace_cache = rows
        return self._trace_cache
