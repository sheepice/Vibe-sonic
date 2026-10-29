"""Serve calibrated D435i COLOR pixels to official ViBe vision_encoders TCP.

Uses its unchanged u32 little-endian length + PNG payload contract. Unlike the
upstream convenience script, this never stops a Unitree service automatically.
"""
import argparse
import io
import socket
import struct
import threading
import time

from PIL import Image

from .camera import RealSenseCamera


def encode_packet(bgr):
    data = io.BytesIO()
    Image.fromarray(bgr[..., ::-1]).save(data, format="PNG", compress_level=1)
    payload = data.getvalue()
    return struct.pack("<I", len(payload)) + payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--serial")
    args = p.parse_args()
    cam = RealSenseCamera(args.serial)
    stop = threading.Event()
    def send(client):
        previous = None
        with client:
            client.settimeout(.1)
            try:
                while not stop.is_set():
                    if cam.frame is None:
                        stop.wait(.01)
                        continue
                    bgr, stamp, frame = cam.latest()
                    if time.monotonic() - stamp > .15:
                        return  # never label stale pixels as a fresh frame
                    if frame != previous:
                        client.sendall(encode_packet(bgr))
                        previous = frame
                    stop.wait(.005)
            except (OSError, RuntimeError):
                return
    try:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.bind, args.port))
            server.listen(4)
            server.settimeout(.2)
            print(f"D435i RGB -> {args.bind}:{args.port}; 112x63 FOV 42.5; no depth", flush=True)
            while not stop.is_set():
                if cam.error:
                    raise RuntimeError(str(cam.error))
                try:
                    client, _ = server.accept()
                except TimeoutError:
                    continue
                threading.Thread(target=send, args=(client,), daemon=True).start()
    finally:
        stop.set()
        cam.close()


if __name__ == "__main__":
    main()
