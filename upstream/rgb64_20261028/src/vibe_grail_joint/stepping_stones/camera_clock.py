"""Causal camera frame holding on the simulator's control-time grid."""
import math


class CameraClock:
    def __init__(self, fps, control_dt):
        if not math.isfinite(fps) or not 0 < fps <= 1/control_dt:
            raise ValueError('Camera FPS must be positive and no faster than the simulation control grid')
        self.fps, self.control_dt = fps, control_dt
        self.slot = -1
        self.captured_at = None
        self.last_time = -math.inf
        self.frames = self.reused = 0
        self.max_age = 0.

    def tick(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed <= self.last_time:
            raise ValueError('Camera ticks require strictly increasing nonnegative simulation time')
        self.last_time = elapsed
        slot = math.floor(elapsed*self.fps+1e-9)
        fresh = slot != self.slot
        if fresh:
            self.slot = slot
            # Capture from the CURRENT simulated image; do not pretend it was
            # observed earlier at the ideal continuous camera deadline.
            self.captured_at = elapsed
            self.frames += 1
        else:
            self.reused += 1
        self.max_age = max(self.max_age, elapsed-self.captured_at)
        return fresh, self.captured_at

    def summary(self):
        return dict(requested_fps=self.fps, encoded_frames=self.frames,
                    reused_control_steps=self.reused, max_image_age_s=self.max_age,
                    capture_time_quantization_s=self.control_dt,
                    additional_transport_delay_s=0.,
                    source='latest causally captured RGB, held between camera updates')
