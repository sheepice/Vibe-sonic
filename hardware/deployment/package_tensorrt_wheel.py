"""Package NVIDIA's privately compiled Python bindings, using installed TRT libs."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build', 'staging', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    source, build, staging, output = [x.resolve() for x in (a.source,a.build,a.staging,a.output)]
    if staging.exists():
        raise FileExistsError(staging)
    header = (source/'include/NvInferVersion.h').read_text()
    version = '.'.join(re.search(r'#define\s+NV_TENSORRT_'+part+r'\s+(\d+)',header).group(1)
                       for part in ('MAJOR','MINOR','PATCH'))
    shutil.copytree(source/'python/packaging/bindings_wheel', staging)
    for path in staging.rglob('*'):
        if path.is_file():
            path.write_text(path.read_text().replace('##TENSORRT_MODULE##','tensorrt')
                            .replace('##TENSORRT_PYTHON_VERSION##',version)
                            .replace('##TENSORRT_VERSION##',version)
                            .replace('##TENSORRT_MAJMINPATCH##',version))
    shutil.copy2(build/'tensorrt/tensorrt.so', staging/'tensorrt/tensorrt.so')
    setup = staging/'setup.py'
    content = setup.read_text().replace('from setuptools import setup',
        'from setuptools import setup, Distribution\n'
        'class BinaryDistribution(Distribution):\n'
        '    def has_ext_modules(self): return True\n')
    setup.write_text(content.replace('setup(\n','setup(\n    distclass=BinaryDistribution,\n',1))
    output.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-m','build','--wheel','--no-isolation',
                    '--outdir',str(output),str(staging)],check=True)


if __name__ == '__main__':
    main()
