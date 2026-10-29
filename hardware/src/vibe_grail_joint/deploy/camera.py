"""D435i COLOR stream -> training pinhole camera; no depth enters the policy."""
import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np


def pinhole_rays(width=112, height=63, fovy=42.5):
    focal = height / (2 * np.tan(np.deg2rad(fovy / 2)))
    y, x = np.mgrid[:height, :width]
    return np.stack([(x + .5 - width / 2) / focal, (y + .5 - height / 2) / focal, np.ones_like(x)], -1)


def bilinear(image, coordinates):
    if (image.ndim != 3 or image.shape[2] != 3 or coordinates.shape[-1] != 2
            or not np.isfinite(coordinates).all()):
        raise ValueError("Invalid camera image or projection coordinates")
    x, y = coordinates[..., 0], coordinates[..., 1]
    if x.min() < 0 or y.min() < 0 or x.max() >= image.shape[1]-1 or y.max() >= image.shape[0]-1:
        raise ValueError("D435i color FOV does not cover the training camera; no black padding allowed")
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    a, b = (x-ix)[..., None], (y-iy)[..., None]
    value = (image[iy, ix] * (1-a)*(1-b) + image[iy, ix+1] * a*(1-b)
             + image[iy+1, ix] * (1-a)*b + image[iy+1, ix+1] * a*b)
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


class RealSenseCamera:
    def __init__(self, serial=None, fovy=42.5, *, resolution=(1280, 720), width=112, height=63,
                 on_frame=None):
        if (width, height) not in ((112, 63), (112, 64)):
            raise ValueError('Unsupported policy camera resolution')
        import pyrealsense2 as rs
        self.rs = rs
        self.on_frame = on_frame
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, *resolution, rs.format.bgr8, 30)
        profile = self.pipeline.start(cfg)
        sensor = profile.get_device().first_color_sensor()
        if sensor.supports(rs.option.frames_queue_size):
            sensor.set_option(rs.option.frames_queue_size, 1)
        if sensor.supports(rs.option.global_time_enabled):
            # We estimate a device-to-monotonic offset below. Do not feed that
            # estimator SDK global timestamps whose host-clock correction moves.
            sensor.set_option(rs.option.global_time_enabled,0)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                               ppx=intr.ppx, ppy=intr.ppy, model=str(intr.model), coeffs=list(intr.coeffs))
        self.serial = profile.get_device().get_info(rs.camera_info.serial_number)
        rays = pinhole_rays(width=width, height=height, fovy=fovy)
        self.mapping = np.array([rs.rs2_project_point_to_pixel(intr, r.tolist()) for r in rays.reshape(-1, 3)]).reshape(height, width, 2)
        self.frame = None
        self.timestamp_domain = None
        self.error = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="d435i-color", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            previous = None
            clock_offset = float("inf")
            clock_ready_deadline = time.monotonic()+2.
            while not self.stop.is_set():
                ok, frames = self.pipeline.try_wait_for_frames(timeout_ms=100)
                if not ok:
                    continue
                color = frames.get_color_frame()
                if not color or color.get_frame_number() == previous:
                    continue
                domain = color.get_frame_timestamp_domain()
                if domain != self.rs.timestamp_domain.hardware_clock:
                    # A frame queued before disabling global time can still
                    # carry the old domain during initial startup.
                    if previous is None and time.monotonic()<clock_ready_deadline:
                        continue
                    raise RuntimeError("D435i hardware frame timestamps are unavailable")
                self.timestamp_domain = str(domain)
                previous = color.get_frame_number()
                received = time.monotonic()
                device_time = color.get_timestamp() / 1000
                clock_offset = min(clock_offset, received - device_time)
                # Timestamp offset estimated from the least delayed arrival, plus frame ID.
                captured = min(received, device_time + clock_offset)
                raw = np.asanyarray(color.get_data())
                bgr = bilinear(raw, self.mapping)
                if self.on_frame:
                    self.on_frame(raw, bgr, captured, previous, device_time)
                with self.lock:
                    self.frame = (bgr, captured, previous)
        except Exception as exc:
            self.error = exc

    def latest(self):
        if self.error:
            raise RuntimeError(f"D435i failed: {self.error}") from self.error
        with self.lock:
            if self.frame is None:
                raise RuntimeError("Waiting for first D435i color frame")
            return self.frame[0].copy(), self.frame[1], self.frame[2]

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.)
        self.pipeline.stop()


def validate_mount(path, camera, expected):
    from .observations import rotation
    data = json.loads(Path(path).read_text())
    if data.get("serial") != camera.serial or data.get("measurement_source") != "measured_on_robot":
        raise ValueError("Camera calibration must identify this device and measured mount")
    if data.get("parent_body") != "torso_link":
        raise ValueError("Camera mount must be measured relative to torso_link")
    pos = np.asarray(data["pos"], dtype=float)
    if pos.shape != (3,) or not np.isfinite(pos).all():
        raise ValueError("Camera position must contain three finite coordinates")
    delta = np.linalg.norm(pos - np.asarray(expected["pos"]))
    angle = (rotation(expected["quat"]).inv() * rotation(data["quat"])).magnitude()
    if delta > .02 or angle > np.deg2rad(5):
        raise ValueError(f"Mount differs from training: {delta:.3f} m, {np.rad2deg(angle):.1f} deg; remount or retrain")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial")
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, default=10.)
    p.add_argument("--bundle", help="Use this model's camera height, width and FOV")
    args = p.parse_args()
    from PIL import Image
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    camera = json.loads((Path(args.bundle)/'bundle.json').read_text())['camera'] if args.bundle else {}
    cam = RealSenseCamera(args.serial, fovy=camera.get('fovy', 42.5),
                          width=camera.get('width', 112), height=camera.get('height', 63))
    try:
        times, ids = [], []
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            if cam.error:
                raise RuntimeError(str(cam.error))
            if cam.frame:
                bgr, stamp, number = cam.latest()
                if not ids or number != ids[-1]:
                    times.append(stamp)
                    ids.append(number)
                    Image.fromarray(bgr[..., ::-1]).save(out / "policy_rgb.png")
            time.sleep(.01)
        if len(ids) < 2:
            raise RuntimeError("No usable camera stream")
        report = dict(serial=cam.serial, color_intrinsics=cam.intrinsics,
                      timestamp_domain=cam.timestamp_domain,
                      frames=len(ids), observed_fps=(len(ids)-1)/(times[-1]-times[0]),
                      input_to_encoder=f"BGR uint8 {bgr.shape[1]}x{bgr.shape[0]}", policy_depth_input=False)
        (out / "camera_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        cam.close()


if __name__ == "__main__":
    main()
