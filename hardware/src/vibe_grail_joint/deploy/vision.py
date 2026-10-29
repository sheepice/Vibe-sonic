"""Encode each fresh camera frame outside the 50 Hz policy loop.

Published features retain their source frame's timestamp. Encoder latency never
refreshes the camera watchdog, and failures propagate to the controller.
"""
import threading


class VisualWorker:
    def __init__(self, runtime, camera, on_features=None):
        self.runtime, self.camera = runtime, camera
        self.on_features = on_features
        self.frame = None
        self.error = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name='visual-encoder', daemon=True)
        self.thread.start()

    def _run(self):
        previous = None
        try:
            while not self.stop.is_set():
                if self.camera.error:
                    raise RuntimeError(f'D435i failed: {self.camera.error}')
                if self.camera.frame is None:
                    self.stop.wait(.002)
                    continue
                image, captured, number = self.camera.latest()
                if number == previous:
                    self.stop.wait(.002)
                    continue
                features = self.runtime.encode(image)
                for value in features:
                    value.flags.writeable = False
                if self.on_features:
                    self.on_features(features, captured, number)
                with self.lock:
                    self.frame = (features, captured, number)
                previous = number
        except Exception as exc:
            self.error = exc

    def latest(self):
        if self.error:
            raise RuntimeError(f'Visual encoder failed: {self.error}') from self.error
        with self.lock:
            if self.frame is None:
                raise RuntimeError('Waiting for first encoded D435i frame')
            return self.frame

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.)
