#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read YOLO detection results (JSON lines) from a FIFO/file and control UR5e via RTDE (speedL).
Expected JSON per line (minimal):
  {"cx": 312, "cy": 214, "conf": 0.83, "w": 120, "h": 80, "cls": "card", "t": 1700.12}

Fastest setup: write these JSON lines into /tmp/yolo.pipe (FIFO).
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Optional, TextIO

try:
    from rtde_control import RTDEControlInterface as RTDEControl
except Exception as e:
    raise SystemExit(
        "ur-rtde not installed or import failed. Install with:\n"
        "  pip install ur-rtde\n"
        f"Error: {e}"
    )


@dataclass
class Detection:
    cx: float
    cy: float
    conf: float
    t: float


class EMAFilter:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._x: Optional[float] = None

    def update(self, v: float) -> float:
        if self._x is None:
            self._x = v
        else:
            self._x = self.alpha * self._x + (1.0 - self.alpha) * v
        return self._x


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def read_detection_line(f: TextIO) -> Optional[Detection]:
    line = f.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
        cx = float(d["cx"])
        cy = float(d["cy"])
        conf = float(d.get("conf", 0.0))
        tt = float(d.get("t", time.time()))
        return Detection(cx=cx, cy=cy, conf=conf, t=tt)
    except Exception:
        # Ignore malformed line
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", required=True, help="UR5e IP address")
    ap.add_argument("--in", dest="in_path", default="/tmp/yolo.pipe", help="FIFO/file path with JSON lines")
    ap.add_argument("--w", type=int, default=640, help="image width used by YOLO")
    ap.add_argument("--h", type=int, default=480, help="image height used by YOLO")

    # Control gains / safety
    ap.add_argument("--k", type=float, default=0.0005, help="pixel->m/s gain (start small)")
    ap.add_argument("--vmax", type=float, default=0.05, help="max linear speed (m/s)")
    ap.add_argument("--acc", type=float, default=0.5, help="acc for speedL")
    ap.add_argument("--dt", type=float, default=0.05, help="control period seconds (e.g., 0.05=20Hz)")
    ap.add_argument("--deadband", type=float, default=10.0, help="deadband in pixels")
    ap.add_argument("--conf-th", type=float, default=0.5, help="min confidence to track")
    ap.add_argument("--timeout", type=float, default=0.25, help="watchdog timeout seconds without detections")
    ap.add_argument("--ema", type=float, default=0.7, help="EMA alpha (0~1), larger=more smoothing")

    # Axis mapping: if robot moves opposite, flip sign here
    ap.add_argument("--flip-x", action="store_true", help="flip vx sign")
    ap.add_argument("--flip-y", action="store_true", help="flip vy sign")

    args = ap.parse_args()

    cx_f = EMAFilter(alpha=args.ema)
    cy_f = EMAFilter(alpha=args.ema)

    # Connect to robot
    rtde_c = RTDEControl(args.robot_ip)
    print(f"[INFO] Connected to UR at {args.robot_ip}")

    # Open FIFO/file (blocking until writer opens FIFO)
    print(f"[INFO] Opening input: {args.in_path}")
    with open(args.in_path, "r", buffering=1) as f:
        last_ok_time = time.time()
        last_rx_time = time.time()

        try:
            while True:
                t0 = time.time()

                det = read_detection_line(f)
                now = time.time()

                if det is not None:
                    last_rx_time = now

                    if det.conf >= args.conf_th:
                        # Filter
                        cx = cx_f.update(det.cx)
                        cy = cy_f.update(det.cy)

                        # Pixel error (center)
                        ex = cx - (args.w / 2.0)
                        ey = cy - (args.h / 2.0)

                        # Deadband
                        if abs(ex) < args.deadband:
                            ex = 0.0
                        if abs(ey) < args.deadband:
                            ey = 0.0

                        # Convert to velocities (m/s)
                        vx = -args.k * ex
                        vy = -args.k * ey

                        if args.flip_x:
                            vx = -vx
                        if args.flip_y:
                            vy = -vy

                        # Clamp speeds
                        vx = clamp(vx, -args.vmax, args.vmax)
                        vy = clamp(vy, -args.vmax, args.vmax)

                        # Move only X/Y (base linear), keep Z and rotations 0
                        # NOTE: If directions feel "rotated", camera is angled; keep low speed or add 4-point homography later.
                        rtde_c.speedL([vx, vy, 0.0, 0.0, 0.0, 0.0], args.acc, args.dt)
                        last_ok_time = now
                    else:
                        # Low confidence => stop
                        rtde_c.speedStop()
                else:
                    # No new line: check watchdog
                    if (now - last_rx_time) > args.timeout:
                        rtde_c.speedStop()

                # Maintain loop timing (best effort)
                elapsed = time.time() - t0
                sleep_t = args.dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C received. Stopping robot.")
        finally:
            try:
                rtde_c.speedStop()
            except Exception:
                pass
            try:
                rtde_c.stopScript()
            except Exception:
                pass
            print("[INFO] Clean exit.")


if __name__ == "__main__":
    main()