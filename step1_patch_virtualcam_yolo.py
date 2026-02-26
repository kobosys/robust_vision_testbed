#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_patch_virtualcam_yolo.py

- Reads frames from a V4L2 camera (e.g., /dev/video10)
- Optionally overlays a patch image (PNG with alpha recommended)
- Runs Ultralytics YOLO inference on each frame
- Optionally shows window / saves video / saves per-frame labels(txt)

Key fix:
- camera index is --camera
- yolo compute device is --yolo-device (cpu, 0, 0,1,2...)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()

    # Camera (V4L2) input
    p.add_argument("--camera", type=int, default=10, help="V4L2 camera index. e.g. 10 => /dev/video10")
    p.add_argument("--width", type=int, default=1280, help="Capture width request")
    p.add_argument("--height", type=int, default=720, help="Capture height request")
    p.add_argument("--fps", type=int, default=30, help="Capture FPS request")

    # YOLO model & inference
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO .pt")
    p.add_argument("--yolo-device", type=str, default="cpu", help="cpu or CUDA device id(s): 0 or 0,1")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    p.add_argument("--classes", type=str, default="", help="Filter classes (comma). e.g. '0,2,3' or empty for all")
    p.add_argument("--max-det", type=int, default=300, help="Max detections per frame")

    # Patch overlay (optional)
    p.add_argument("--patch", type=str, default="", help="Path to patch image (PNG recommended)")
    p.add_argument("--patch-x", type=int, default=50, help="Patch top-left x (pixels)")
    p.add_argument("--patch-y", type=int, default=50, help="Patch top-left y (pixels)")
    p.add_argument("--patch-scale", type=float, default=1.0, help="Patch scale factor")
    p.add_argument("--patch-alpha", type=float, default=1.0, help="Global patch alpha multiplier (0~1)")

    # Output
    p.add_argument("--show", action="store_true", help="Show preview window")
    p.add_argument("--save-video", action="store_true", help="Save annotated video")
    p.add_argument("--save-txt", action="store_true", help="Save YOLO-format labels per frame (runs/labels/*.txt)")
    p.add_argument("--save-dir", type=str, default="runs/patch_virtualcam", help="Output directory")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=until quit)")

    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_patch(patch_path: str) -> Optional[np.ndarray]:
    if not patch_path:
        return None
    img = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)  # keep alpha if exists
    if img is None:
        raise FileNotFoundError(f"Failed to read patch image: {patch_path}")
    return img


def overlay_patch(
    frame_bgr: np.ndarray,
    patch: np.ndarray,
    x: int,
    y: int,
    scale: float = 1.0,
    alpha_mul: float = 1.0,
) -> np.ndarray:
    """Overlay patch onto frame at (x, y). Patch may be BGR or BGRA."""
    out = frame_bgr.copy()

    if patch is None:
        return out

    if scale <= 0:
        return out

    ph, pw = patch.shape[:2]
    new_w = max(1, int(pw * scale))
    new_h = max(1, int(ph * scale))
    patch_rs = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)

    H, W = out.shape[:2]

    # compute ROI bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + new_w), min(H, y + new_h)
    if x1 >= x2 or y1 >= y2:
        return out

    roi = out[y1:y2, x1:x2]

    # patch crop corresponding to ROI
    px1, py1 = x1 - x, y1 - y
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
    patch_crop = patch_rs[py1:py2, px1:px2]

    if patch_crop.shape[2] == 4:
        # BGRA
        patch_bgr = patch_crop[:, :, :3].astype(np.float32)
        a = (patch_crop[:, :, 3].astype(np.float32) / 255.0) * float(np.clip(alpha_mul, 0.0, 1.0))
        a = a[:, :, None]  # (h,w,1)
    else:
        # BGR without alpha -> use global alpha
        patch_bgr = patch_crop.astype(np.float32)
        a = np.ones((patch_crop.shape[0], patch_crop.shape[1], 1), dtype=np.float32) * float(np.clip(alpha_mul, 0.0, 1.0))

    roi_f = roi.astype(np.float32)
    blended = roi_f * (1.0 - a) + patch_bgr * a
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def draw_results(frame: np.ndarray, results) -> np.ndarray:
    # results[0].plot() returns BGR annotated image
    try:
        return results[0].plot()
    except Exception:
        return frame


def to_class_list(s: str) -> Optional[List[int]]:
    if not s.strip():
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if not parts:
        return None
    return [int(x) for x in parts]


def save_yolo_txt(labels_dir: Path, frame_idx: int, results) -> None:
    """
    Save YOLO labels for one frame:
    class x_center y_center width height (normalized 0~1)
    """
    labels_path = labels_dir / f"{frame_idx:06d}.txt"

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        labels_path.write_text("")  # create empty file for traceability
        return

    # xywhn: normalized xywh
    xywhn = boxes.xywhn.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    lines = []
    for i in range(len(cls)):
        c = cls[i]
        x, y, w, h = xywhn[i]
        lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    labels_path.write_text("\n".join(lines) + "\n")


def open_camera(camera_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index={camera_index} (try /dev/video{camera_index}).")

    # best-effort property set
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    labels_dir = save_dir / "labels"
    if args.save_txt:
        ensure_dir(labels_dir)

    # Load model
    model = YOLO(args.model)

    # Load patch (optional)
    patch = read_patch(args.patch) if args.patch else None

    # Camera
    cap = open_camera(args.camera, args.width, args.height, args.fps)

    # Video writer (optional)
    writer = None
    if args.save_video:
        out_path = save_dir / "out.mp4"
        # read actual capture size
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps is None or actual_fps <= 1:
            actual_fps = args.fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(actual_fps), (actual_w, actual_h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {out_path}")

    classes = to_class_list(args.classes)

    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Failed to read frame. Exiting.")
                break

            # Apply patch overlay if requested
            if patch is not None:
                frame_in = overlay_patch(
                    frame,
                    patch,
                    x=args.patch_x,
                    y=args.patch_y,
                    scale=args.patch_scale,
                    alpha_mul=args.patch_alpha,
                )
            else:
                frame_in = frame

            # YOLO inference
            results = model.predict(
                source=frame_in,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.yolo_device,   # ✅ compute device (cpu / 0 / 0,1...)
                classes=classes,
                max_det=args.max_det,
                verbose=False,
            )

            # Draw annotated frame
            annotated = draw_results(frame_in, results)

            # Save labels
            if args.save_txt:
                save_yolo_txt(labels_dir, frame_idx, results)

            # Save video
            if writer is not None:
                writer.write(annotated)

            # Show
            if args.show:
                cv2.imshow("patch_virtualcam_yolo", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

        dt = time.time() - t0
        if dt > 0 and frame_idx > 0:
            print(f"[INFO] Done. frames={frame_idx}, avg_fps={frame_idx/dt:.2f}")
        print(f"[INFO] Output dir: {save_dir.resolve()}")


if __name__ == "__main__":
    main()