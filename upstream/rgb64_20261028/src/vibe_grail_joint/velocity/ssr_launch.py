"""SSR now launches the integrated memory policy on physical GPUs 0-3."""
import argparse
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('/mnt/workspace/wjw/GRAIL/ssr_memory_20261023')
    p = argparse.ArgumentParser()
    p.add_argument('--gpus', choices=('0', '0,1,2,3'), default='0,1,2,3')
    p.add_argument('--config', default=str(root / 'configs/train.json'))
    p.add_argument('--output', '--log-dir', dest='output')
    p.add_argument('--updates', '--iterations', dest='updates', type=int)
    args, rest = p.parse_known_args()
    if Path(args.config).resolve() != root / 'configs/train.json':
        raise ValueError('Legacy SSR configs are retired. Use ssr_memory_20261023/configs/train.json.')
    output = args.output or str(root / 'runs' / ('ssr_memory_' + time.strftime('%Y%m%d_%H%M%S')))
    command = [sys.executable, str(root / 'launch.py'), '--gpus', args.gpus,
               '--config', args.config, '--output', output, *rest]
    if args.updates is not None:
        command += ['--updates', str(args.updates)]
    raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == '__main__':
    main()
