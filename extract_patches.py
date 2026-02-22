#!/usr/bin/env python3
"""
Extract recognizable patches from an image using a pretrained classifier.

Slides windows across the source image at multiple scales, runs each patch
through a MobileNetV2 classifier, and saves patches where the top-1
classification confidence is high — meaning the model recognizes a clear
subject in that crop.

Goal: build a folder of recognizable patches to later match against
larger target images with search_caltech_rembg.py.

Usage:
    python extract_patches.py photo.jpg
    python extract_patches.py photo.jpg --threshold 0.3 --max_patches 20
    python extract_patches.py photo.jpg --patch_sizes 96 160 224 --stride 0.4
"""

import os
import math
import argparse
import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

_model = None
_INPUT_SIZE = 224


def get_classifier():
    """Load MobileNetV2 pretrained on ImageNet (cached after first call)."""
    global _model
    if _model is None:
        print("Loading MobileNetV2 classifier...")
        _model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return _model


def classify_patch(patch_rgb, model):
    """Run a single patch through the classifier.

    Returns (score, label) where score is the top-1 softmax probability
    and label is the predicted ImageNet class name.
    """
    resized = cv2.resize(patch_rgb, (_INPUT_SIZE, _INPUT_SIZE))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        resized.astype(np.float32)[np.newaxis]
    )
    preds = model(x, training=False).numpy()
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
    _, label, confidence = decoded[0]
    return float(confidence), label


def extract_patches(image_path, output_dir, patch_sizes=(96, 160, 224),
                    stride_fraction=0.5, threshold=0.15, max_patches=50):
    """Extract recognizable patches from an image.

    Args:
        image_path: Path to the source image.
        output_dir: Folder to save accepted patches into.
        patch_sizes: Window sizes to try (pixels).
        stride_fraction: Stride as fraction of patch size (0.5 = 50% overlap).
        threshold: Minimum classifier confidence to keep a patch.
        max_patches: Maximum number of patches to save.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]

    os.makedirs(output_dir, exist_ok=True)
    model = get_classifier()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    candidates = []

    print(f"Scanning {image_path} ({img_w}x{img_h}) with patch sizes {patch_sizes}")

    for patch_size in patch_sizes:
        if patch_size > min(img_h, img_w):
            continue

        stride = max(16, int(patch_size * stride_fraction))
        n_y = len(range(0, img_h - patch_size + 1, stride))
        n_x = len(range(0, img_w - patch_size + 1, stride))

        print(f"  Patch size {patch_size}: {n_y * n_x} windows (stride {stride})")

        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                patch = img_rgb[y:y + patch_size, x:x + patch_size]
                score, label = classify_patch(patch, model)

                if score >= threshold:
                    candidates.append({
                        'patch': patch,
                        'score': score,
                        'label': label,
                        'x': x, 'y': y,
                        'size': patch_size,
                    })

    # Sort by score descending
    candidates.sort(key=lambda c: c['score'], reverse=True)

    # Deduplicate overlapping patches: skip if center is inside an already-kept patch
    kept = []
    for c in candidates:
        cx, cy = c['x'] + c['size'] // 2, c['y'] + c['size'] // 2
        overlap = False
        for k in kept:
            if (k['x'] <= cx < k['x'] + k['size'] and
                    k['y'] <= cy < k['y'] + k['size']):
                overlap = True
                break
        if not overlap:
            kept.append(c)
        if len(kept) >= max_patches:
            break

    # Save patches
    for i, c in enumerate(kept):
        safe_label = c['label'].replace(' ', '_')
        fname = f"{base_name}_p{i:03d}_{safe_label}_{c['score']:.2f}.png"
        out_path = os.path.join(output_dir, fname)
        patch_bgr = cv2.cvtColor(c['patch'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, patch_bgr)

    print(f"\nKept {len(kept)} / {len(candidates)} candidate patches (threshold {threshold})")
    print(f"Saved to: {output_dir}/")

    if kept:
        _visualize_kept(img_rgb, kept, output_dir, base_name)

    return kept


def _visualize_kept(img_rgb, kept, output_dir, base_name):
    """Save a visualization showing kept patches on the source image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Overview: source image with rectangles
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_rgb)
    for c in kept:
        rect = plt.Rectangle((c['x'], c['y']), c['size'], c['size'],
                              linewidth=1.5, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(c['x'] + 2, c['y'] + 12,
                f"{c['label']} {c['score']:.2f}",
                color='lime', fontsize=6, weight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))
    ax.set_title(f"Kept patches ({len(kept)})")
    ax.axis("off")
    overview_path = os.path.join(output_dir, f"{base_name}_overview.png")
    plt.tight_layout()
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Overview saved: {overview_path}")

    # Grid of extracted patches
    n = len(kept)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array(axes)[np.newaxis, :]
    elif cols == 1:
        axes = np.array(axes)[:, np.newaxis]
    for i, c in enumerate(kept):
        r, col_idx = divmod(i, cols)
        axes[r, col_idx].imshow(c['patch'])
        axes[r, col_idx].set_title(f"{c['label']}\n{c['score']:.2f}", fontsize=7)
        axes[r, col_idx].axis("off")
    for i in range(n, rows * cols):
        r, col_idx = divmod(i, cols)
        axes[r, col_idx].axis("off")
    grid_path = os.path.join(output_dir, f"{base_name}_grid.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grid saved: {grid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract recognizable patches from an image using a classifier")
    parser.add_argument("image", help="Path to source image")
    parser.add_argument("--output_dir", default="image_patches",
                        help="Output folder for patches (default: image_patches)")
    parser.add_argument("--patch_sizes", nargs="+", type=int, default=[96, 160, 224],
                        help="Patch window sizes in pixels (default: 96 160 224)")
    parser.add_argument("--stride", type=float, default=0.5,
                        help="Stride as fraction of patch size (default: 0.5)")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Minimum classifier confidence to keep (default: 0.15)")
    parser.add_argument("--max_patches", type=int, default=50,
                        help="Maximum patches to save (default: 50)")

    args = parser.parse_args()

    extract_patches(
        args.image,
        output_dir=args.output_dir,
        patch_sizes=args.patch_sizes,
        stride_fraction=args.stride,
        threshold=args.threshold,
        max_patches=args.max_patches,
    )
