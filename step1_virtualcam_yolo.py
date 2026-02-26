#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_virtualcam_yolo.py
- 가상카메라(/dev/video10 등)에서 프레임을 읽어 YOLO 추론
- 가장 conf 높은 1개 탐지 결과(cx, cy, conf, t)를 /tmp/yolo.pipe 로 JSON Lines로 출력

터미널 B에서 실행 예:
  mkfifo /tmp/yolo.pipe
  python3 step1_virtualcam_yolo.py --device /dev/video10 --model yolov8n.pt --show

주의:
- FIFO는 리더(UR 제어 step2)가 열기 전엔 writer가 block될 수 있어
  이 코드는 non-blocking으로 열고, 리더 없으면 출력만 스킵합니다.
"""

import argparse
import json
import os
import time
from typing import Optional, Dict, Any

import cv2

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit(
        "Ultralytics YOLO가 필요합니다.\n"
        "  pip install ultralytics\n"
        f"Error: {e}"
    )


def ensure_fifo(path: str):
    if os.path.exists(path):
        # 파일/파이프가 이미 있으면 그대로 사용
        return
    os.mkfifo(path)


class FifoWriter:
    """리더가 없을 때도 프로그램이 멈추지 않도록 non-blocking FIFO writer."""
    def __init__(self, path: str):
        self.path = path
        self.fd: Optional[int] = None

    def try_open(self):
        if self.fd is not None:
            return
        try:
            # 리더가 없으면 ENXIO가 발생할 수 있음
            self.fd = os.open(self.path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError:
            self.fd = None

    def write_line(self, s: str):
        if self.fd is None:
            self.try_open()
            if self.fd is None:
                return  # 리더 아직 없음 -> 출력 스킵
        try:
            os.write(self.fd, (s + "\n").encode("utf-8"))
        except OSError:
            # 리더가 끊기면 다시 열도록
            try:
                os.close(self.fd)
            except Exception:
                pass
            self.fd = None

    def close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                pass
            self.fd = None


def best_detection_from_ultralytics(result, conf_th: float) -> Optional[Dict[str, Any]]:
    """
    result: Ultralytics Results 1개(프레임 1장에 대한 결과)
    반환: {"cx":..., "cy":..., "conf":..., "cls":..., "w":..., "h":...} 또는 None
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes
    # xyxy: (N,4), conf: (N,), cls: (N,)
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    # conf 조건 만족하는 것 중 최대값
    best_i = -1
    best_c = conf_th
    for i, c in enumerate(confs):
        if c >= best_c:
            best_c = float(c)
            best_i = i

    if best_i < 0:
        return None

    x1, y1, x2, y2 = map(float, xyxy[best_i])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1)
    h = (y2 - y1)
    cls_id = int(clss[best_i])

    return {"cx": cx, "cy": cy, "conf": best_c, "cls": cls_id, "w": w, "h": h, "xyxy": (x1, y1, x2, y2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video10", help="입력 카메라 디바이스 (예: /dev/video10)")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO 모델 경로")
    ap.add_argument("--imgsz", type=int, default=640, help="추론 입력 크기")
    ap.add_argument("--conf-th", type=float, default=0.5, help="최소 conf (이하 출력/표시 안 함)")
    ap.add_argument("--fifo", default="/tmp/yolo.pipe", help="탐지 결과 출력 FIFO 경로")
    ap.add_argument("--show", action="store_true", help="OpenCV 창으로 프리뷰 표시")
    ap.add_argument("--width", type=int, default=640, help="캡처 폭(가능하면 지정)")
    ap.add_argument("--height", type=int, default=480, help="캡처 높이(가능하면 지정)")
    ap.add_argument("--fps", type=int, default=30, help="캡처 FPS(가능하면 지정)")
    args = ap.parse_args()

    # FIFO 준비
    ensure_fifo(args.fifo)
    fifo = FifoWriter(args.fifo)

    # 카메라 오픈
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"카메라를 열 수 없습니다: {args.device}")

    # 캡처 파라미터 세팅(환경에 따라 무시될 수 있음)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # 실제 캡처 크기 확인
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    print(f"[INFO] Capture: {W}x{H} @ {cap.get(cv2.CAP_PROP_FPS):.1f} fps")
    print(f"[INFO] FIFO out: {args.fifo} (reader 없으면 출력은 자동 스킵)")
    print("[INFO] Press 'q' to quit (when --show).")

    # YOLO 로드
    model = YOLO(args.model)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # 추론
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf_th, verbose=False)
            result = results[0] if results else None

            det = best_detection_from_ultralytics(result, conf_th=args.conf_th)

            # 결과 FIFO로 출력: 가장 conf 높은 1개만
            if det is not None:
                payload = {
                    "cx": det["cx"],
                    "cy": det["cy"],
                    "conf": float(det["conf"]),
                    "cls": det["cls"],
                    "w": det["w"],
                    "h": det["h"],
                    "t": time.time(),
                }
                fifo.write_line(json.dumps(payload, ensure_ascii=False))

            # 프리뷰
            if args.show:
                vis = frame
                if det is not None:
                    x1, y1, x2, y2 = det["xyxy"]
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(vis, (int(det["cx"]), int(det["cy"])), 5, (0, 0, 255), -1)
                    cv2.putText(
                        vis,
                        f"cls={det['cls']} conf={det['conf']:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("YOLO VirtualCam", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    finally:
        fifo.close()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[INFO] Exit.")


if __name__ == "__main__":
    main()