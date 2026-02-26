import time
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs
import pyfakewebcam


def apply_corruption(frame_bgr: np.ndarray, corrupt: str, severity: int) -> np.ndarray:
    if corrupt == "none" or severity <= 0:
        return frame_bgr

    out = frame_bgr.copy()

    if corrupt == "gaussian_noise":
        sigma_list = [5, 10, 15, 25, 35]
        sigma = sigma_list[min(max(severity, 1), 5) - 1]
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out

    if corrupt == "gaussian_blur":
        k_list = [3, 5, 7, 9, 11]
        k = k_list[min(max(severity, 1), 5) - 1]
        out = cv2.GaussianBlur(out, (k, k), 0)
        return out

    if corrupt == "jpeg":
        q_list = [80, 60, 45, 30, 15]
        q = q_list[min(max(severity, 1), 5) - 1]
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return out

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dev/video10", help="v4l2loopback device, e.g. /dev/video10")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--corrupt", default="none", choices=["none", "gaussian_noise", "gaussian_blur", "jpeg"])
    ap.add_argument("--severity", type=int, default=0, help="1~5")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Run time in seconds (0 = infinite). Example: --duration 60",
    )
    args = ap.parse_args()

    # RealSense setuppython step1_virtualcam_yolo.py
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline.start(config)

    # v4l2loopback output via pyfakewebcam (raw frames)
    # NOTE: pyfakewebcam expects RGB frames (uint8)
    cam = pyfakewebcam.FakeWebcam(args.out, args.width, args.height)

    if args.preview:
        cv2.namedWindow("Corrupt Proxy Preview", cv2.WINDOW_NORMAL)

    print(f"[OK] Proxy running: RealSense -> {args.corrupt}(sev={args.severity}) -> {args.out}")
    if args.duration > 0:
        print(f"Auto-stop after {args.duration} seconds.")
    else:
        print("Run indefinitely. (Ctrl+C to stop)")
    print("Press Ctrl+C to stop. (ESC to exit preview window)")

    start_time = time.time()

    try:
        while True:
            # Auto-stop condition
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print("[OK] Duration reached. Stopping...")
                break

            frameset = pipeline.wait_for_frames()
            color = frameset.get_color_frame()
            if not color:
                continue

            frame_bgr = np.asanyarray(color.get_data())
            frame_cor_bgr = apply_corruption(frame_bgr, args.corrupt, args.severity)

            # Send to virtual camera (RGB)
            frame_rgb = cv2.cvtColor(frame_cor_bgr, cv2.COLOR_BGR2RGB)
            cam.schedule_frame(frame_rgb)

            # Optional preview
            if args.preview:
                cv2.imshow("Corrupt Proxy Preview", frame_cor_bgr)
                if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                    print("[OK] ESC pressed. Stopping...")
                    break

    except KeyboardInterrupt:
        print("\n[OK] KeyboardInterrupt. Stopping...")
    finally:
        pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()