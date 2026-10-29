"""Private fast LowCmd CDR packing; SDK classes and wire format stay unchanged.

Only the pinned final G1 schema is optimized. Every factory initialization
checks both CDR versions against CycloneDDS using a nonzero probe message.
No DDS entity is constructed here.
"""
import struct
from types import MethodType
import weakref

_V1 = struct.Struct('<3Bx5fI'+'B3x5fI'*34+'5I')
_V2 = struct.Struct('<2B2xI'+'B3x5fI'*35+'5I')


def serialize_lowcmd(message,buffer=None,endianness=None,use_version_2=None):
    if use_version_2 is None:
        # CycloneDDS DataWriter normally passes None (datatype default).
        # Resolve the version verified against the SDK by this factory, so
        # production sends use the same packing path as explicit-version tests.
        use_version_2 = getattr(message, '_vibe_default_version_2', None)
    if buffer is not None or endianness is not None or use_version_2 not in (False,True):
        return message.__class__.serialize(message,buffer=buffer,endianness=endianness,use_version_2=use_version_2)
    if len(message.motor_cmd)!=35 or len(message.reserve)!=4:
        raise ValueError('Invalid G1 LowCmd array length')
    values = [message.mode_pr,message.mode_machine]
    if use_version_2:
        values.append(980)  # XCDR2 DHEADER for the 35 fixed MotorCmd elements.
    for motor in message.motor_cmd:
        values.extend((motor.mode,motor.q,motor.dq,motor.tau,motor.kp,motor.kd,motor.reserve))
    values.extend(message.reserve)
    values.append(message.crc)
    return (b'\x00\x07\x00\x00'+_V2.pack(*values) if use_version_2 else
            b'\x00\x01\x00\x00'+_V1.pack(*values))


def lowcmd_factory():
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as factory
    probe = factory()
    if getattr(probe,'__idl_typename__',None)!='unitree_hg.msg.dds_.LowCmd_':
        return factory  # Fake/unknown SDK: preserve its implementation.
    probe.mode_pr,probe.mode_machine = 1,7
    probe.reserve,probe.crc = [9,10,11,12],13
    for i,motor in enumerate(probe.motor_cmd):
        motor.mode,motor.reserve = i%2,i+100
        motor.q,motor.dq,motor.tau,motor.kp,motor.kd = i+.125,-i-.25,i+.5,30.,2.
    versions = {}
    for version in (False,True):
        versions[version] = probe.serialize(use_version_2=version)
        if serialize_lowcmd(probe,use_version_2=version)!=versions[version]:
            raise RuntimeError('G1 CDR layout differs from validated serializer')
    default_bytes = probe.serialize(use_version_2=None)
    default_version = next((v for v, data in versions.items() if data == default_bytes), None)
    if default_version is None:
        raise RuntimeError('G1 default CDR version differs from validated serializer')
    def create():
        message = factory()
        message._vibe_default_version_2 = default_version
        # Binding to the object itself forms 500 reference cycles per second,
        # causing large cyclic-GC pauses. A proxy keeps immediate refcount cleanup.
        message.serialize = MethodType(serialize_lowcmd,weakref.proxy(message))
        return message
    return create
