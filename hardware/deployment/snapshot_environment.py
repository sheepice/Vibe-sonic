"""Record private runtime provenance; no robot/camera SDK initialization."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path(__file__).resolve().parents[2]
out = root/'records/deployment_20261019/optimization_environment.json'
if out.exists():
    raise FileExistsError(out)
def git_head(directory):
    return subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
files = [*root.glob('models/velocity3999/*.onnx'),*root.glob('models/velocity3999/*.json'),
         *root.glob('engines/velocity3999_trt10_fp32/*'),*root.glob('wheelhouse/*.whl')]
headers = root/'third_party/TensorRT-10.3/python/include/onnx'
files += [headers/name for name in ('NvOnnxParser.h','errorHelpers.hpp','Status.hpp')]
report = dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
    source_commits={name:git_head(root/'third_party'/name) for name in
                    ('unitree_sdk2_python','librealsense-2.58.1','TensorRT-10.3')},
    parser_helper_commit='62bdde2a04fcd53c2409cb895ee18db445b7e755',
    files_sha256={str(p.relative_to(root)):digest(p) for p in files if p.is_file()},
    private_camera_backend='V4L2, existing kernel driver; static private librealsense',
    original_drivers_preserved=True,system_cuda='12.6',system_tensorrt='10.3.0',
    hardware_control_test=False,camera_mount_measured=False,physical_stop_test=False,
    clocks='Temporary comparison tests only; original settings restored after tests')
out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(out)
