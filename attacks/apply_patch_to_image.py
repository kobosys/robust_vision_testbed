import cv2
import argparse
import numpy as np


def apply_patch(image_path, patch_path, output_path, x, y):
    image = cv2.imread(image_path)
    patch = cv2.imread(patch_path)

    h, w = patch.shape[:2]

    image[y:y+h, x:x+w] = patch

    cv2.imwrite(output_path, image)
    print(f"[INFO] Saved patched image to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--x", type=int, default=50)
    parser.add_argument("--y", type=int, default=50)

    args = parser.parse_args()

    apply_patch(
        args.image,
        args.patch,
        args.output,
        args.x,
        args.y
    )


if __name__ == "__main__":
    main()