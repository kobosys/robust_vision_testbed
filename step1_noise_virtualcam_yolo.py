import cv2
import time
from ultralytics import YOLO

VIDEO_INDEX = 10        # /dev/video10
WIDTH = 640
HEIGHT = 480
FPS = 30
MODEL_PATH = "yolov8n.pt"

cap = cv2.VideoCapture(VIDEO_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError(f"/dev/video{VIDEO_INDEX} 열기 실패")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

model = YOLO(MODEL_PATH)

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("frame read failed")
        time.sleep(0.05)
        continue

    results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = model.names.get(cls, str(cls))

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            print(f"{name:>12} conf={conf:.2f} center=({cx:.0f},{cy:.0f})")

            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    current_time = time.time()
    fps = 1.0 / max(current_time - prev_time, 1e-6)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("VirtualCam -> YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()