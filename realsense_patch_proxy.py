import cv2
import numpy as np
import argparse
import pyrealsense2 as rs
import random

def overlay_patch(frame, patch, scale=0.2, pos_x=0.5, pos_y=0.5, rotate=True):
    h, w, _ = frame.shape

    # 크기 조정
    patch_w = int(w * scale)
    ratio = patch_w / patch.shape[1]
    patch_resized = cv2.resize(patch, None, fx=ratio, fy=ratio)

    # 회전(EOT 간단버전)
    if rotate:
        angle = random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D(
            (patch_resized.shape[1] // 2, patch_resized.shape[0] // 2),
            angle,
            1
        )
        patch_resized = cv2.warpAffine(
            patch_resized,
            M,
            (patch_resized.shape[1], patch_resized.shape[0])
        )

    ph, pw, _ = patch_resized.shape

    x = int(w * pos_x - pw / 2)
    y = int(h * pos_y - ph / 2)

    x = max(0, min(w - pw, x))
    y = max(0, min(h - ph, y))

    roi = frame[y:y+ph, x:x+pw]

    # 알파채널 지원
    if patch_resized.shape[2] == 4:
        alpha = patch_resized[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (
                alpha * patch_resized[:, :, c] +
                (1 - alpha) * roi[:, :, c]
            )
    else:
        roi[:] = patch_resized[:, :, :3]

    frame[y:y+ph, x:x+pw] = roi
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", required=True)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--pos_x", type=float, default=0.5)
    parser.add_argument("--pos_y", type=float, default=0.7)
    parser.add_argument("--out", default="/dev/video10")
    args = parser.parse_args()

    patch = cv2.imread(args.patch, cv2.IMREAD_UNCHANGED)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    out = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*'YUYV'),
        30,
        (640, 480)
    )

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        frame = overlay_patch(frame, patch, args.scale, args.pos_x, args.pos_y)

        out.write(frame)

        cv2.imshow("Patch Proxy Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()