#!/usr/bin/env python3
"""
Train (optimize) an adversarial patch against a surrogate YOLO model (Ultralytics YOLOv8/YOLOv5 style),
by directly optimizing the patch tensor with PyTorch.

This is an "ART-style" patch optimization loop:
- EOT-like random transforms (scale/rotation/brightness/contrast + optional blur)
- Optimize patch to reduce detection confidence (untargeted)

Requirements:
  pip install ultralytics torch opencv-python numpy

Example:
  python attacks/train_patch_art.py \
    --weights yolov8n.pt \
    --images data/patch_train_images \
    --out patches/trained/trained_patch.png \
    --iters 400 --batch 4 --patch_size 256

Notes:
- Ultralytics model forward output shapes differ by version.
  We compute a generic "confidence proxy" from raw outputs and minimize its top-k average.
"""

import argparse
import os
from pathlib import Path
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics is required. Install with: pip install ultralytics") from e


def load_images(folder: str, img_size: int, max_images: int = 2000) -> list[np.ndarray]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Images folder not found: {folder}")

    files = [x for x in sorted(p.rglob("*")) if x.suffix.lower() in exts]
    if not files:
        raise RuntimeError(f"No images found in: {folder}")

    files = files[:max_images]
    imgs = []
    for f in files:
        bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        bgr = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        imgs.append(bgr)
    if not imgs:
        raise RuntimeError("Failed to load any images (all reads returned None).")
    return imgs


def bgr_to_tensor01(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    # bgr uint8 HWC -> float32 NCHW in [0,1]
    x = torch.from_numpy(bgr).to(device=device, dtype=torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return x


def batch_from_pool(pool: list[np.ndarray], batch: int, device: torch.device) -> torch.Tensor:
    chosen = random.sample(pool, k=min(batch, len(pool)))
    xs = [bgr_to_tensor01(im, device) for im in chosen]
    return torch.cat(xs, dim=0)  # B,3,H,W


def tv_loss(patch: torch.Tensor) -> torch.Tensor:
    # patch: 1,3,H,W
    dh = torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :]).mean()
    dw = torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1]).mean()
    return dh + dw


def apply_eot_transform(patch: torch.Tensor,
                        out_h: int,
                        out_w: int,
                        scale_min: float,
                        scale_max: float,
                        rot_deg: float,
                        brightness: float,
                        contrast: float,
                        blur: bool) -> torch.Tensor:
    """
    patch: 1,3,Ph,Pw in [0,1]
    returns: 1,3,h,w transformed patch (h,w vary by scale) -> we will resize to final (h,w) via grid_sample.
    """
    device = patch.device
    _, _, ph, pw = patch.shape

    # Random scale (relative)
    s = float(torch.empty(1).uniform_(scale_min, scale_max).item())

    # Random rotation
    ang = float(torch.empty(1).uniform_(-rot_deg, rot_deg).item()) * np.pi / 180.0
    ca, sa = np.cos(ang), np.sin(ang)

    # Build affine matrix for grid_sample
    # grid_sample expects theta in normalized coordinates
    theta = torch.tensor([[[ca * s, -sa * s, 0.0],
                           [sa * s,  ca * s, 0.0]]],
                         device=device, dtype=torch.float32)

    grid = F.affine_grid(theta, size=(1, 3, out_h, out_w), align_corners=False)
    warped = F.grid_sample(patch, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Photometric (approx EOT)
    if contrast > 0:
        c = float(torch.empty(1).uniform_(1.0 - contrast, 1.0 + contrast).item())
        warped = warped * c
    if brightness > 0:
        b = float(torch.empty(1).uniform_(-brightness, brightness).item())
        warped = warped + b
    warped = warped.clamp(0.0, 1.0)

    if blur:
        # Small differentiable blur (avg pool)
        warped = F.avg_pool2d(warped, kernel_size=3, stride=1, padding=1)

    return warped


def paste_patch(images: torch.Tensor,
                patch_t: torch.Tensor,
                alpha: float,
                pos_mode: str) -> torch.Tensor:
    """
    images: B,3,H,W in [0,1]
    patch_t: 1,3,H,W in [0,1] (already transformed to full H,W canvas with zeros elsewhere)
    alpha: 0..1
    pos_mode: center or random (random is done via shifting patch canvas)
    """
    B, _, H, W = images.shape
    device = images.device

    # Build patch canvas with random shift (translation) using torch.roll on the already-canvas-sized patch
    if pos_mode == "random":
        # shift range so patch can appear anywhere
        # (roll is differentiable w.r.t patch values)
        sx = int(torch.randint(low=-W // 3, high=W // 3 + 1, size=(1,), device=device).item())
        sy = int(torch.randint(low=-H // 3, high=H // 3 + 1, size=(1,), device=device).item())
        patch_canvas = torch.roll(patch_t, shifts=(sy, sx), dims=(2, 3))
    else:
        patch_canvas = patch_t

    # Alpha blend: image*(1-a*m) + patch*(a*m)
    # Here we use mask where patch is non-zero (approx)
    mask = (patch_canvas.sum(dim=1, keepdim=True) > 1e-6).float()  # 1,1,H,W
    mask = mask.expand(B, 1, H, W)

    patch_canvas_b = patch_canvas.expand(B, 3, H, W)
    out = images * (1.0 - alpha * mask) + patch_canvas_b * (alpha * mask)
    return out.clamp(0.0, 1.0)


def yolo_confidence_proxy(raw_out: torch.Tensor) -> torch.Tensor:
    """
    Produce a generic confidence score tensor from raw model outputs.
    Works as a proxy even if exact YOLO head differs by version.

    Expected raw_out shapes vary:
      - (B, N, D) or (B, D, N)
    We assume class logits start at index 4 (x,y,w,h) or are the only logits.
    """
    if raw_out.dim() != 3:
        # some versions return list/tuple - caller should handle
        raise RuntimeError(f"Unexpected model output dim: {raw_out.shape}")

    B = raw_out.shape[0]

    # Put as (B, N, D)
    if raw_out.shape[1] < raw_out.shape[2]:
        # likely (B, N, D)
        pred = raw_out
    else:
        # likely (B, D, N)
        pred = raw_out.permute(0, 2, 1)

    D = pred.shape[-1]

    # Heuristic:
    # - if D >= 6: treat pred[...,4:] as class/objectness-related logits
    # - if D <= 5: treat everything as logits
    if D >= 6:
        logits = pred[..., 4:]
    else:
        logits = pred

    probs = torch.sigmoid(logits)
    # confidence proxy per candidate = max class prob
    conf = probs.max(dim=-1).values  # (B, N)
    return conf


@torch.no_grad()
def save_patch_png(patch01: torch.Tensor, out_path: str):
    """
    patch01: 1,3,H,W in [0,1]
    """
    p = patch01.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0
    bgr = p[..., ::-1].astype(np.uint8)  # RGB->BGR
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = cv2.imwrite(out_path, bgr)
    if not ok:
        raise RuntimeError(f"Failed to write: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="yolov8n.pt", help="Ultralytics YOLO weights path")
    ap.add_argument("--images", type=str, required=True, help="Folder with training images (jpg/png)")
    ap.add_argument("--img_size", type=int, default=640, help="Model input size (square)")
    ap.add_argument("--patch_size", type=int, default=256, help="Patch base size (square)")
    ap.add_argument("--iters", type=int, default=400, help="Optimization steps")
    ap.add_argument("--batch", type=int, default=4, help="Batch size per step")
    ap.add_argument("--lr", type=float, default=0.10, help="Adam learning rate for patch")
    ap.add_argument("--alpha", type=float, default=1.0, help="Overlay strength 0..1")
    ap.add_argument("--topk", type=int, default=200, help="Top-k detections to penalize")
    ap.add_argument("--tv", type=float, default=0.01, help="Total-variation regularization weight")
    ap.add_argument("--scale_min", type=float, default=0.15, help="EOT scale min (relative)")
    ap.add_argument("--scale_max", type=float, default=0.35, help="EOT scale max (relative)")
    ap.add_argument("--rot_deg", type=float, default=25.0, help="EOT rotation degrees")
    ap.add_argument("--brightness", type=float, default=0.15, help="EOT brightness jitter (0..)")
    ap.add_argument("--contrast", type=float, default=0.20, help="EOT contrast jitter (0..)")
    ap.add_argument("--blur", action="store_true", help="Enable simple blur EOT")
    ap.add_argument("--pos", choices=["center", "random"], default="random", help="Patch placement mode")
    ap.add_argument("--out", type=str, default="patches/trained/trained_patch.png", help="Output patch PNG")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")

    # Load images pool
    pool = load_images(args.images, img_size=args.img_size)
    print(f"[INFO] loaded {len(pool)} images from: {args.images}")

    # Load YOLO model (surrogate)
    y = YOLO(args.weights)
    model = y.model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Patch parameter (optimize this)
    patch = torch.rand(1, 3, args.patch_size, args.patch_size, device=device, dtype=torch.float32, requires_grad=True)

    opt = torch.optim.Adam([patch], lr=args.lr)

    H = W = args.img_size

    # Center patch canvas (H,W) with patch placed in the middle (then we roll for random placement)
    def make_center_canvas(transformed_patch_full: torch.Tensor) -> torch.Tensor:
        # transformed_patch_full is already 1,3,H,W with zeros where outside
        return transformed_patch_full

    for step in range(1, args.iters + 1):
        x = batch_from_pool(pool, args.batch, device)  # B,3,H,W in [0,1]

        # Make a full-canvas transformed patch using affine warp into (H,W)
        patch_clamped = patch.clamp(0.0, 1.0)
        patch_full = apply_eot_transform(
            patch_clamped,
            out_h=H,
            out_w=W,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            rot_deg=args.rot_deg,
            brightness=args.brightness,
            contrast=args.contrast,
            blur=args.blur
        )  # 1,3,H,W

        patched = paste_patch(x, patch_full, alpha=float(np.clip(args.alpha, 0.0, 1.0)), pos_mode=args.pos)

        # Forward
        out = model(patched)

        # Normalize output extraction
        if isinstance(out, (list, tuple)):
            raw = out[0]
        else:
            raw = out

        conf = yolo_confidence_proxy(raw)  # (B, N)
        k = min(args.topk, conf.shape[1])
        topk_mean = conf.topk(k, dim=1).values.mean()

        # Objective: reduce confidence (minimize)
        loss_main = topk_mean
        loss_tv = tv_loss(patch_clamped)
        loss = loss_main + args.tv * loss_tv

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Keep patch bounded
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)

        if step == 1 or step % 25 == 0:
            print(f"[{step:04d}/{args.iters}] loss={loss.item():.4f} "
                  f"(main={loss_main.item():.4f}, tv={loss_tv.item():.4f})")

        # Save intermediate snapshots
        if step % 100 == 0 or step == args.iters:
            save_patch_png(patch, args.out)
            print(f"[SAVE] {args.out}")

    print("[DONE] Patch optimization finished.")
    print(f"       Patch saved at: {args.out}")


if __name__ == "__main__":
    main()