"""Compare the hardware input/action chain with the tested sim2sim oracle.

Synthetic states and RGB exercise heading, commands, histories and reset.
No robot SDK, camera device or motor transport is imported.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim2sim.vendor.observations import ObservationBuilder as SimBuilder
from sim2sim.vendor.runtime import Runtime as SimRuntime
from vibe_grail_joint.deploy.observations import ObservationBuilder, Reference, RobotState
from vibe_grail_joint.deploy.runtime import Runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    real, sim = Runtime(a.bundle), SimRuntime(a.bundle)
    builders = [ObservationBuilder(real.contract, real.patch_shape),
                SimBuilder(sim.contract, sim.patch_shape)]
    rng = np.random.default_rng(20261028)
    errors = {}
    def compare(name, x, y):
        assert x.shape == y.shape and np.isfinite([x, y]).all(), name
        error = float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))))
        errors[name] = max(errors.get(name, 0.), error)
        np.testing.assert_allclose(x, y, atol=1e-6, rtol=0, err_msg=name)
    commands = [(0., 0.), (.4, 0.), (.4, .45), (0., -.45), (-.1, 0.), (0., 0.)]
    heading = 0.
    for tick in range(72):
        if tick in (0, 36):
            for b in builders:
                b.reset()
        speed, yaw = commands[tick//12]
        heading += yaw*.02
        state = RobotState(real.default+rng.normal(0,.05,29).astype(np.float32),
            rng.normal(0,.1,29).astype(np.float32),
            Rotation.from_euler('xyz', [.03, -.04, heading+.1]).as_quat()[[3,0,1,2]],
            rng.normal(0,.05,3).astype(np.float32), tick*.02)
        ref = Reference(np.tile(real.default,(10,1))+rng.normal(0,.03,(10,29)),
            rng.normal(0,.1,(10,29)),
            np.tile(Rotation.from_euler('z',heading).as_quat()[[3,0,1,2]],(10,1)),
            np.array([speed*np.cos(heading),speed*np.sin(heading),0.]), np.array([0.,0.,yaw]))
        image = rng.integers(0,256,real.image_shape,dtype=np.uint8)
        real_features, sim_features = real.encode(image), sim.encode(image)
        for name, x, y in zip(('patches','cls'), real_features, sim_features, strict=True):
            compare(name, x, y)
        feeds = [b.build(state,ref,*features,speed,heading=heading,yaw_rate=yaw)
                 for b,features in zip(builders,(real_features,sim_features),strict=True)]
        for name in feeds[0]:
            compare('input/'+name, feeds[0][name], feeds[1][name])
        outputs = real.act(feeds[0]), sim.act(feeds[1])
        compare('action', outputs[0][0], outputs[1][0])
        compare('pd_target', outputs[0][1], outputs[1][1])
        for b, (action, _, _) in zip(builders,outputs,strict=True):
            b.accept_action(action)
    old = json.loads((a.bundle.parent/'velocity3999/bundle.json').read_text())
    assert old['robot'] == real.contract
    assert old['files']['planner.onnx'] == real.manifest['files']['planner.onnx']
    report = dict(passed=True,cases=72,history_reset_ticks=[0,36],commands=commands,
        bundle_files=real.manifest['files'],checkpoint_sha256=real.manifest['checkpoint_sha256'],
        image_shape=real.image_shape,patch_shape=real.patch_shape,max_absolute_error=errors,
        previous_robot_contract_identical=True,previous_planner_identical=True,
        motor_transport=False,real_motor_test=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    real.close()


if __name__ == '__main__':
    main()
