"""Check model-dependent camera projection without opening a USB device."""
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from vibe_grail_joint.deploy import camera


@pytest.mark.parametrize('height', [63, 64])
def test_realsense_maps_to_training_grid(monkeypatch, height):
    intr = NS(width=1280,height=720,fx=800.,fy=800.,ppx=640.,ppy=360.,model='none',coeffs=[0.]*5)
    sensor = NS(supports=lambda _: False)
    profile = NS(get_device=lambda: NS(first_color_sensor=lambda:sensor,get_info=lambda _:'fake'),
                 get_stream=lambda _: NS(as_video_stream_profile=lambda:NS(get_intrinsics=lambda:intr)))
    pipeline = NS(start=lambda _:profile,stop=lambda:None)
    config = NS(enable_device=lambda _:None,enable_stream=lambda *a:None)
    fake_rs = NS(pipeline=lambda:pipeline,config=lambda:config,stream=NS(color=1),format=NS(bgr8=1),
        option=NS(frames_queue_size=1,global_time_enabled=2),camera_info=NS(serial_number=1),
        rs2_project_point_to_pixel=lambda i,r:[i.fx*r[0]+i.ppx,i.fy*r[1]+i.ppy])
    monkeypatch.setitem(sys.modules,'pyrealsense2',fake_rs)
    monkeypatch.setattr(camera.threading,'Thread',lambda **k:NS(start=lambda:None,join=lambda **k:None))
    cam = camera.RealSenseCamera('fake',width=112,height=height)
    try:
        rays = camera.pinhole_rays(width=112,height=height)
        np.testing.assert_allclose(cam.mapping, rays[:,:,:2]*800.+[640.,360.])
        assert cam.mapping.shape == (height,112,2)
        frame = camera.bilinear(np.zeros((720,1280,3),np.uint8),cam.mapping)
        assert frame.shape == (height,112,3)
        # Pixel-center geometry must remain symmetric for both even/odd heights.
        np.testing.assert_allclose(rays[:,:,1], -rays[::-1,:,1])
    finally:
        cam.close()
