import os
import argparse
import numpy as np
import cv2


def generate_random_patch(width, height, save_path):
    patch = np.random.randint(
        0, 256, (height, width, 3), dtype=np.uint8
    )

    cv2.imwrite(save_path, patch)
    print(f"[INFO] Saved: {save_path}")
    print(f"[INFO] Size: {width} x {height}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate random RGB patch"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=100,
        help="Patch width (pixels)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=100,
        help="Patch height (pixels)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="patches/random/random_patch.png",
        help="Output file path"
    )

    args = parser.parse_args()

    generate_random_patch(
        args.width,
        args.height,
        args.output
    )


if __name__ == "__main__":
    main()