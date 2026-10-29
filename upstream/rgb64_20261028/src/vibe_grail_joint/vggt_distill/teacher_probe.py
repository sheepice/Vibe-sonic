"""Validate actual local checkpoint, batch isolation, patch targets and memory."""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    import torch
    from .teacher import GeometryTeacher
    torch.set_num_threads(2)
    settings = json.loads(Path(args.config).read_text())['distillation']
    start = time.monotonic()
    teacher = GeometryTeacher(settings, 'cuda:0')
    images = torch.randint(0, 255, (2, 64, 112, 3), dtype=torch.uint8)
    result = teacher(images)
    separate = teacher(images[:1])
    error = float((separate - result[:1]).abs().max())
    changed = images.clone(); changed[1].zero_()
    invariant = teacher(changed)
    leakage = float((invariant[:1] - result[:1]).abs().max())
    if result.shape != (2, 3, 28, 2048) or not torch.isfinite(result).all() or error > .015 or leakage > 1e-6:
        raise RuntimeError(f'Teacher extraction or batch isolation failed: {result.shape}, {error}')
    teacher.assert_frozen()
    record = dict(shape=list(result.shape), singleton_batch_numerical_error=error,
                  cross_world_perturbation_error=leakage, frozen=True,
                  seconds=time.monotonic() - start, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  target_spatial_std=float(result.float().std(dim=2).mean()))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__': main()
