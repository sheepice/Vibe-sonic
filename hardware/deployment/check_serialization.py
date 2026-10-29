"""Compare optimized message bytes to official CDR; no DDS entities."""
import argparse
import json
from pathlib import Path
import time
import weakref
import numpy as np
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from vibe_grail_joint.deploy.serialization import lowcmd_factory
from vibe_grail_joint.deploy.crc import optimized_crc

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
factory=lowcmd_factory()
probe=factory()
reference=weakref.ref(probe)
del probe
assert reference() is None, 'Serializer retains a cycle requiring garbage collection'
rng=np.random.default_rng(20261019)
durations={'official':[],'optimized':[]}
crc=optimized_crc()
for case in range(512):
    message=factory()
    message.mode_pr,message.mode_machine=case%2,case%256
    message.reserve=rng.integers(0,2**32,4,dtype=np.uint32).tolist()
    for motor in message.motor_cmd:
        motor.mode=int(rng.integers(0,256))
        motor.reserve=int(rng.integers(0,2**32,dtype=np.uint32))
        for field in ('q','dq','tau','kp','kd'):
            setattr(motor,field,float(rng.normal()))
    message.crc=crc.Crc(message)
    for version in (None,False,True):
        start=time.monotonic()
        expected=type(message).serialize(message,use_version_2=version)
        durations['official'].append(time.monotonic()-start)
        start=time.monotonic()
        actual=message.serialize(use_version_2=version)
        durations['optimized'].append(time.monotonic()-start)
        assert actual==expected,(case,version)
        restored=type(message).deserialize(actual)
        assert crc.Crc(restored)==restored.crc
report=dict(passed=True,messages=512,byte_comparisons=1536,cdr_versions=['default',1,2],hardware_control_test=False,
            timing_us={k:dict(p50=float(np.percentile(v,50)*1e6),p99=float(np.percentile(v,99)*1e6))
                       for k,v in durations.items()})
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
