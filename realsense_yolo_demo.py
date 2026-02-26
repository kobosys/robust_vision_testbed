#!/usr/bin/env python3
"""
RealSense + YOLOv8 demo -> "고정된 실험" 하네스 버전

추가/개선 사항
- 고정 실행: --duration_sec 또는 --max_frames
- 실험 폴더 자동 생성: save_dir 아래에 YYYYMMDD_HHMMSS 폴더 생성
- 결과 CSV 로깅: metrics.csv (프레임별 1줄)
- 설정 저장: settings.yaml
- 저장 옵션 분리: 원본(raw) / 주석(annotated)
- 추론 옵션 고정: conf, iou, imgsz, max_det, classes
- CPU 강제 및 CUDA 실패시 자동 폴백 유지
"""

import os
import time
import argparse
from datetime import datetime
from typing import Optional, List

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO


def pick_device(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    return "cuda:0"


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def parse_classes(s: Optional[str]) -> Optional[List[int]]:
    """
    --classes "0,1,2" 형태를 [0,1,2]로 변환.
    None이면 전체 클래스 사용.
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def write_settings_yaml(path: str, args: argparse.Namespace, device: str):
    # yaml 라이브러리 없이 최소한의 YAML 텍스트로 저장
    lines = []
    lines.append(f"created_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"model: {args.model}")
    lines.append(f"device: {device}")
    lines.append(f"conf: {args.conf}")
    lines.append(f"iou: {args.iou}")
    lines.append(f"imgsz: {args.imgsz}")
    lines.append(f"max_det: {args.max_det}")
    lines.append(f"classes: {args.classes if args.classes is not None else 'ALL'}")
    lines.append(f"width: {args.width}")
    lines.append(f"height: {args.height}")
    lines.append(f"fps: {args.fps}")
    lines.append(f"duration_sec: {args.duration_sec if args.duration_sec > 0 else 'OFF'}")
    lines.append(f"max_frames: {args.max_frames if args.max_frames > 0 else 'OFF'}")
    lines.append(f"save_every_sec: {args.save_every if args.save_every > 0 else 'OFF'}")
    lines.append(f"save_raw: {bool(args.save_raw)}")
    lines.append(f"save_annotated: {bool(args.save_annotated)}")
    lines.append(f"preview: {bool(args.preview)}")
    lines.append(f"save_video: {bool(args.save_video)}")
    lines.append(f"video_fps: {args.video_fps}")
    content = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def open_csv(path: str):
    f = open(path, "w", encoding="utf-8", newline="")
    header = (
        "timestamp_ms,frame_id,detected,conf_max,"
        "bbox_x1,bbox_y1,bbox_x2,bbox_y2,"
        "bbox_cx,bbox_cy,bbox_w,bbox_h,"
        "cls_id,dist_m,fps_est\n"
    )
    f.write(header)
    f.flush()
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path (e.g., yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--max_det", type=int, default=50, help="Max detections per image")
    parser.add_argument("--classes", default=None, help='Filter classes, e.g. "0,1,2" (default: ALL)')
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    # RealSense 고정
    parser.add_argument("--width", type=int, default=640, help="RealSense color/depth width")
    parser.add_argument("--height", type=int, default=480, help="RealSense color/depth height")
    parser.add_argument("--fps", type=int, default=30, help="RealSense FPS")

    # 실험 고정(종료 조건)
    parser.add_argument("--duration_sec", type=int, default=60, help="Stop after N seconds (0=off)")
    parser.add_argument("--max_frames", type=int, default=0, help="Stop after N frames (0=off). If set, wins over duration.")

    # 저장/출력
    parser.add_argument(
        "--save_dir",
        default="/mnt/datasets/robust_yolo/runs",
        help="Base directory for experiments (a timestamped subdir will be created)"
    )
    parser.add_argument("--save_every", type=int, default=0, help="Auto save snapshot every N seconds (0=off)")
    parser.add_argument("--save_raw", action="store_true", help="Save raw frames (no drawings)")
    parser.add_argument("--save_annotated", action="store_true", help="Save annotated frames (with bbox)")
    parser.add_argument("--save_video", action="store_true", help="Save annotated video (mp4)")
    parser.add_argument("--video_fps", type=int, default=30, help="FPS for saved video")

    # 미리보기(화면 출력)
    parser.add_argument("--preview", action="store_true", help="Show preview window")

    args = parser.parse_args()
    cls_filter = parse_classes(args.classes)

    # -------------------------
    # 실험 폴더 생성 (timestamp)
    # -------------------------
    exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.save_dir, exp_name)
    frames_dir = os.path.join(exp_dir, "frames")
    raw_dir = os.path.join(frames_dir, "raw")
    ann_dir = os.path.join(frames_dir, "annotated")
    logs_dir = os.path.join(exp_dir, "logs")

    ensure_dir(exp_dir)
    ensure_dir(logs_dir)
    ensure_dir(frames_dir)
    if args.save_raw:
        ensure_dir(raw_dir)
    if args.save_annotated or args.save_video:
        ensure_dir(ann_dir)

    # 설정 저장
    device = pick_device(args.cpu)
    write_settings_yaml(os.path.join(exp_dir, "settings.yaml"), args, device)

    # CSV 로그
    csv_path = os.path.join(exp_dir, "metrics.csv")
    csv_f = open_csv(csv_path)

    # -------------------------
    # RealSense pipeline
    # -------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # depth scale (참고용: depth_frame.get_distance()는 meters)
    depth_sensor = profile.get_device().first_depth_sensor()
    _ = depth_sensor.get_depth_scale()

    # -------------------------
    # YOLO model
    # -------------------------
    model = YOLO(args.model)

    # -------------------------
    # Preview / Video writer
    # -------------------------
    if args.preview:
        cv2.namedWindow("RealSense + YOLO (Experiment)", cv2.WINDOW_NORMAL)

    video_writer = None
    if args.save_video:
        # mp4v 코덱(환경에 따라 H264는 없을 수 있음)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = os.path.join(exp_dir, "annotated.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, args.video_fps, (args.width, args.height))

    # -------------------------
    # Loop
    # -------------------------
    start_time = time.time()
    last_save = 0.0
    frame_id = 0

    # FPS 추정용
    t0 = time.time()
    frames_for_fps = 0
    fps_est = 0.0

    print("======================================================")
    print("Fixed Experiment Runner")
    print(f"EXP_DIR: {exp_dir}")
    print(f"CSV:     {csv_path}")
    print(f"Model: {args.model} | conf={args.conf} | iou={args.iou} | imgsz={args.imgsz} | device={device}")
    print("Stop condition:", end=" ")
    if args.max_frames > 0:
        print(f"max_frames={args.max_frames}")
    elif args.duration_sec > 0:
        print(f"duration_sec={args.duration_sec}")
    else:
        print("OFF (manual stop)")
    if args.preview:
        print("Preview: ON (ESC quit, Space pause/resume, S save snapshot)")
    else:
        print("Preview: OFF")
    if not args.cpu:
        print("NOTE: CUDA(sm_120 등) 호환 문제면 자동으로 CPU로 폴백합니다.")
    print("======================================================")

    paused = False

    try:
        while True:
            # 종료 조건
            if args.max_frames > 0 and frame_id >= args.max_frames:
                break
            if args.max_frames <= 0 and args.duration_sec > 0 and (time.time() - start_time) >= args.duration_sec:
                break

            if args.preview and paused:
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
                if key == ord(" "):
                    paused = False
                continue

            frameset = pipeline.wait_for_frames()
            frameset = align.process(frameset)

            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame_raw = np.asanyarray(color_frame.get_data())
            frame_vis = frame_raw.copy()  # 주석용

            # YOLO 추론 (실패하면 CPU로 폴백)
            try:
                pred = model.predict(
                    frame_raw,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=device,
                    max_det=args.max_det,
                    classes=cls_filter,
                    verbose=False
                )[0]
            except Exception as e:
                if device != "cpu":
                    print(f"[WARN] YOLO CUDA inference failed -> fallback to CPU. Error: {e}")
                    device = "cpu"
                    pred = model.predict(
                        frame_raw,
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        device=device,
                        max_det=args.max_det,
                        classes=cls_filter,
                        verbose=False
                    )[0]
                else:
                    raise

            # FPS 추정
            frames_for_fps += 1
            if frames_for_fps % 30 == 0:
                dt = time.time() - t0
                fps_est = (frames_for_fps / dt) if dt > 0 else 0.0

            # 기본값(탐지 없음)
            detected = 0
            conf_max = 0.0
            bbox = (-1, -1, -1, -1)
            cx, cy, bw, bh = (-1, -1, -1, -1)
            cls_id = -1
            dist_m = -1.0

            # 결과 처리: 여러 박스 중 conf 최대 1개만 "대표값"으로 기록
            if pred.boxes is not None and len(pred.boxes) > 0:
                boxes = pred.boxes
                # conf 최대 인덱스 찾기
                confs = boxes.conf.detach().cpu().numpy()
                best_i = int(np.argmax(confs))
                b = boxes[best_i]

                x1, y1, x2, y2 = b.xyxy[0].detach().cpu().numpy().astype(int)
                cls_id = int(b.cls[0].detach().cpu().numpy())
                conf_max = float(b.conf[0].detach().cpu().numpy())

                x1 = int(np.clip(x1, 0, args.width - 1))
                y1 = int(np.clip(y1, 0, args.height - 1))
                x2 = int(np.clip(x2, 0, args.width - 1))
                y2 = int(np.clip(y2, 0, args.height - 1))

                bbox = (x1, y1, x2, y2)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)

                dist_m = float(depth_frame.get_distance(int(cx), int(cy)))
                detected = 1

                # 시각화
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame_vis, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(
                    frame_vis,
                    f"cls:{cls_id} conf:{conf_max:.2f} dist:{dist_m:.2f}m",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

            # CSV 기록 (프레임마다 1줄)
            ts_ms = int(time.time() * 1000)
            x1, y1, x2, y2 = bbox
            csv_f.write(
                f"{ts_ms},{frame_id},{detected},{conf_max:.4f},"
                f"{x1},{y1},{x2},{y2},"
                f"{cx},{cy},{bw},{bh},"
                f"{cls_id},{dist_m:.4f},{fps_est:.2f}\n"
            )
            if frame_id % 30 == 0:
                csv_f.flush()

            # 저장(자동/수동)
            now = time.time()
            do_auto_save = args.save_every > 0 and (now - last_save) >= args.save_every

            if do_auto_save:
                stamp = f"{ts_ms}"
                if args.save_raw:
                    cv2.imwrite(os.path.join(raw_dir, f"raw_{stamp}.jpg"), frame_raw)
                if args.save_annotated:
                    cv2.imwrite(os.path.join(ann_dir, f"ann_{stamp}.jpg"), frame_vis)
                last_save = now

            # 비디오 저장(annotated)
            if video_writer is not None:
                video_writer.write(frame_vis)

            # 미리보기
            if args.preview:
                # 창 제목에 FPS/디바이스 표시
                cv2.setWindowTitle(
                    "RealSense + YOLO (Experiment)",
                    f"RealSense + YOLO | FPS~{fps_est:.1f} | device={device} | exp={os.path.basename(exp_dir)}"
                )
                cv2.imshow("RealSense + YOLO (Experiment)", frame_vis)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    break
                if key in (ord("s"), ord("S")):
                    stamp = f"{ts_ms}"
                    if args.save_raw:
                        out_raw = os.path.join(raw_dir, f"raw_{stamp}.jpg")
                        cv2.imwrite(out_raw, frame_raw)
                        print(f"[SAVE RAW] {out_raw}")
                    out_ann = os.path.join(ann_dir, f"ann_{stamp}.jpg")
                    cv2.imwrite(out_ann, frame_vis)
                    print(f"[SAVE ANN] {out_ann}")
                if key == ord(" "):
                    paused = True

            frame_id += 1

    finally:
        try:
            csv_f.flush()
            csv_f.close()
        except Exception:
            pass
        if video_writer is not None:
            video_writer.release()
        pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()

    print("======================================================")
    print("Experiment finished.")
    print(f"EXP_DIR: {exp_dir}")
    print(f"CSV:     {csv_path}")
    if args.save_video:
        print(f"VIDEO:   {os.path.join(exp_dir, 'annotated.mp4')}")
    print("======================================================")


if __name__ == "__main__":
    main()