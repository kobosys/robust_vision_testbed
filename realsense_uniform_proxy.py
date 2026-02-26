#!/usr/bin/env python3
import argparse
import os
import sys
import time
import subprocess
import numpy as np
import cv2
import pyrealsense2 as rs


def apply_uniform_noise(frame_bgr: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    if K <= 0:
        return frame_bgr
    noise = rng.integers(-K, K + 1, size=frame_bgr.shape, dtype=np.int16)
    out = frame_bgr.astype(np.int16) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def build_realsense_pipeline(width: int, height: int, fps: int) -> rs.pipeline:
    pipeline = rs.pipeline()
    cfg = rs.config()
    # 호환성 좋은 rgb8로 받고 OpenCV에서 BGR로 변환
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    pipeline.start(cfg)
    return pipeline


def open_v4l2_writer_opencv(path: str, width: int, height: int, fps: int) -> cv2.VideoWriter:
    for fourcc_str in ["MJPG", "YUYV"]:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, cv2.CAP_V4L2, fourcc, float(fps), (width, height), True)
        if writer.isOpened():
            print(f"[INFO] OpenCV writer opened with {fourcc_str} via CAP_V4L2")
            return writer
    raise RuntimeError(
        f"Failed to open v4l2 output with OpenCV: {path}\n"
        f"Try ffmpeg writer (--writer ffmpeg) and check:\n"
        f"  v4l2-ctl -d {path} --list-formats-ext\n"
    )


def open_v4l2_writer_ffmpeg(path: str, width: int, height: int, fps: int, out_pix_fmt: str) -> subprocess.Popen:
    """
    Pipe raw BGR frames (bgr24) into ffmpeg, ffmpeg converts to out_pix_fmt and writes to v4l2.
    out_pix_fmt 추천: yuyv422 (대부분 호환)
    """
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-f", "v4l2",
        "-pix_fmt", out_pix_fmt,
        path,
    ]
    print("[INFO] FFmpeg cmd:", " ".join(cmd))
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    if p.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin pipe.")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dev/video10")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0, help="Use -1 for time-based seed")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no_write", action="store_true")
    ap.add_argument("--max_seconds", type=float, default=0.0)
    ap.add_argument("--writer", choices=["ffmpeg", "opencv"], default="ffmpeg",
                    help="v4l2 output method. ffmpeg is most reliable.")
    ap.add_argument("--out_pix_fmt", default="yuyv422",
                    help="ffmpeg v4l2 output pixel format (e.g., yuyv422, mjpeg if supported)")
    args = ap.parse_args()

    if args.seed == -1:
        seed = int(time.time() * 1000) % (2**31 - 1)
    else:
        seed = args.seed

    rng = np.random.default_rng(seed)

    if not args.no_write:
        if not os.path.exists(args.out):
            raise RuntimeError(f"{args.out} does not exist. Create v4l2loopback device first.")
        if args.writer == "opencv":
            writer = open_v4l2_writer_opencv(args.out, args.width, args.height, args.fps)
            ffmpeg_proc = None
        else:
            writer = None
            ffmpeg_proc = open_v4l2_writer_ffmpeg(args.out, args.width, args.height, args.fps, args.out_pix_fmt)
        print(f"[INFO] Writing to {args.out} via {args.writer}")
    else:
        writer = None
        ffmpeg_proc = None
        print("[INFO] no_write mode: preview only")

    pipeline = build_realsense_pipeline(args.width, args.height, args.fps)
    print(f"[INFO] RealSense started (rgb8 {args.width}x{args.height}@{args.fps})")
    print(f"[INFO] Uniform noise K={args.K}, seed={seed}")

    t0 = time.time()
    frames = 0

    try:
        while True:
            fs = pipeline.wait_for_frames()
            cf = fs.get_color_frame()
            if not cf:
                continue

            frame_rgb = np.asanyarray(cf.get_data())
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            attacked = apply_uniform_noise(frame_bgr, args.K, rng)

            if not args.no_write:
                if args.writer == "opencv":
                    writer.write(attacked)
                else:
                    # ffmpeg stdin에 raw bgr 바이트를 그대로 투입
                    ffmpeg_proc.stdin.write(attacked.tobytes())

            if args.preview:
                cv2.imshow("RealSense Uniform Noise Proxy", attacked)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frames += 1
            if args.max_seconds > 0 and (time.time() - t0) >= args.max_seconds:
                print(f"[INFO] max_seconds reached: {args.max_seconds}")
                break

    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass
            ffmpeg_proc.terminate()
        if args.preview:
            cv2.destroyAllWindows()
        print(f"[INFO] Stopped. frames={frames}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
        sys.exit(0)