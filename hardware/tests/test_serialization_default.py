"""Pinned SDK integration: default DataWriter representation must use fast packing."""
import pytest


def test_default_representation_matches_official_bytes_without_falling_back(monkeypatch):
    pytest.importorskip('unitree_sdk2py.idl.default')
    from vibe_grail_joint.deploy.serialization import lowcmd_factory
    message = lowcmd_factory()()
    message.motor_cmd[3].q = -.137
    message.motor_cmd[3].kp = 27.
    message.mode_machine = 5
    expected = {v: type(message).serialize(message, use_version_2=v) for v in (None, False, True)}
    def forbidden_fallback(*a, **kw):
        raise AssertionError('DataWriter default bypassed fast serializer')
    monkeypatch.setattr(type(message), 'serialize', forbidden_fallback)
    for version in (None, False, True):
        assert message.serialize(use_version_2=version) == expected[version]
