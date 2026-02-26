#!/usr/bin/env python3
import argparse
import subprocess
import sys
import cv2
import numpy as np
import pyrealsense2 as rs


def start_ffmpeg_v4l2_writer(out_dev: str, w: int, h: int, fps: int) -> subprocess.Popen:
    """
    FFmpeg로 stdin(raw BGR24)을 받아 v4l2(/dev/videoX)로 yuyv422로 출력.
    가장 호환성/안정성이 좋음.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-f", "v4l2",
        "-pix_fmt", "yuyv422",
        out_dev,
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다. `sudo apt install -y ffmpeg`")
    if proc.stdin is None:
        raise RuntimeError("FFmpeg stdin 파이프 생성 실패")
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dev/video10", help="v4l2loopback output device")
    ap.add_argument("--patch", default="patches/random/random_patch.png", help="patch image path")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--x", type=int, default=200)
    ap.add_argument("--y", type=int, default=150)
    ap.add_argument("--no-preview", action="store_true", help="disable cv2.imshow preview window")
    args = ap.parse_args()

    # --- Load patch ---
    patch = cv2.imread(args.patch, cv2.IMREAD_COLOR)
    if patch is None:
        raise RuntimeError(f"Patch not found: {args.patch}")
    ph, pw = patch.shape[:2]

    # --- RealSense setup ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    try:
        pipeline.start(config)
    except Exception as e:
        raise RuntimeError(f"RealSense start failed: {e}")

    # --- FFmpeg v4l2 yuyv422 writer ---
    proc = start_ffmpeg_v4l2_writer(args.out, args.width, args.height, args.fps)
    print(f"[INFO] Sending to {args.out} as yuyv422 via ffmpeg")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())  # BGR (H,W,3)

            # Safety resize (혹시라도 들어오는 크기가 다르면 고정)
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

            # Apply patch (simple overwrite)
            x, y = args.x, args.y
            if 0 <= x < args.width and 0 <= y < args.height:
                x2 = min(args.width, x + pw)
                y2 = min(args.height, y + ph)
                patch_crop = patch[: (y2 - y), : (x2 - x)]
                frame[y:y2, x:x2] = patch_crop

            # Write BGR bytes to ffmpeg stdin -> ffmpeg converts to yuyv422 and outputs to v4l2
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                raise RuntimeError("FFmpeg 파이프가 끊어졌습니다. /dev/video10 점유/권한/포맷을 확인하세요.")

            if not args.no_preview:
                cv2.imshow("RealSense + Patch (BGR preview)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        try:
            proc.terminate()
        except Exception:
            pass

        cv2.destroyAllWindows()
        print("[INFO] Exit.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)