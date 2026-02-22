#!/usr/bin/env python3
"""
Extract recognizable patches from an image using GradCAM-guided extraction.

Instead of sliding a window and classifying every crop (slow), this:
  1. Runs the classifier ONCE on the full image to get top predictions
  2. Computes GradCAM heatmaps to locate where each prediction fires
  3. Finds peak regions in the heatmaps
  4. Extracts patches centered on those peaks at multiple scales
  5. Re-classifies each candidate patch to confirm + get final scores

This is orders of magnitude faster than brute-force sliding window.

Goal: build a folder of recognizable patches to later match against
larger target images with search_caltech_rembg.py.

Usage:
    python extract_patches.py photo.jpg
    python extract_patches.py photo.jpg --threshold 0.15 --max_patches 20
    python extract_patches.py photo.jpg --top_classes 10 --patch_scales 0.15 0.25 0.4
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


def gradcam(model, img_tensor, class_index):
    """Compute GradCAM heatmap for a given class.

    Returns a heatmap (H x W) in [0, 1] showing where the class activates.
    """
    # Last conv layer in MobileNetV2
    last_conv = model.get_layer("out_relu")

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(img_tensor, training=False)
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_out)
    # Global average pooling of gradients
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(conv_out * weights, axis=-1).numpy()[0]

    # ReLU + normalize
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam


def find_peaks(heatmap, min_distance=20, threshold_rel=0.3):
    """Find local maxima in a heatmap.

    Returns list of (y, x, intensity) sorted by intensity descending.
    """
    h, w = heatmap.shape
    thresh = threshold_rel * heatmap.max()
    peaks = []

    # Simple non-maximum suppression on a grid
    step = max(1, min_distance // 2)
    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = max(0, y - min_distance)
            y2 = min(h, y + min_distance + 1)
            x1 = max(0, x - min_distance)
            x2 = min(w, x + min_distance + 1)
            local_max = heatmap[y1:y2, x1:x2].max()
            if heatmap[y, x] == local_max and heatmap[y, x] >= thresh:
                peaks.append((y, x, float(heatmap[y, x])))

    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks


def extract_patches(image_path, output_dir, top_classes=10,
                    patch_scales=(0.15, 0.25, 0.4),
                    threshold=0.15, max_patches=50):
    """Extract recognizable patches using GradCAM-guided search.

    Args:
        image_path: Path to the source image.
        output_dir: Folder to save accepted patches into.
        top_classes: Number of top classifier predictions to generate heatmaps for.
        patch_scales: Patch sizes as fractions of the image's shorter side.
        threshold: Minimum classifier confidence to keep a re-classified patch.
        max_patches: Maximum number of patches to save.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]
    short_side = min(img_h, img_w)

    os.makedirs(output_dir, exist_ok=True)
    model = get_classifier()
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Step 1: classify full image once
    resized = cv2.resize(img_rgb, (_INPUT_SIZE, _INPUT_SIZE))
    img_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(
        resized.astype(np.float32)[np.newaxis]
    )
    preds = model(img_tensor, training=False).numpy()
    top_indices = np.argsort(preds[0])[::-1][:top_classes]

    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=top_classes)[0]
    print(f"Top {top_classes} predictions for full image:")
    for _, label, conf in decoded:
        print(f"  {label}: {conf:.3f}")

    # Step 2: compute GradCAM for each top class, find peaks
    candidate_regions = []  # (cy_frac, cx_frac, class_idx, label, heatmap_intensity)

    for rank, class_idx in enumerate(top_indices):
        cam = gradcam(model, img_tensor, int(class_idx))
        peaks = find_peaks(cam, min_distance=3, threshold_rel=0.3)

        cam_h, cam_w = cam.shape
        _, label, _ = decoded[rank]

        for py, px, intensity in peaks[:5]:  # top 5 peaks per class
            cy_frac = py / cam_h
            cx_frac = px / cam_w
            candidate_regions.append((cy_frac, cx_frac, int(class_idx), label, intensity))

    print(f"\nFound {len(candidate_regions)} candidate regions from GradCAM peaks")

    # Step 3: extract patches at each peak at multiple scales, re-classify
    candidates = []
    seen_centers = set()

    for cy_frac, cx_frac, class_idx, cam_label, intensity in candidate_regions:
        cy_px = int(cy_frac * img_h)
        cx_px = int(cx_frac * img_w)

        for scale in patch_scales:
            patch_size = max(32, int(short_side * scale))

            x1 = max(0, cx_px - patch_size // 2)
            y1 = max(0, cy_px - patch_size // 2)
            x2 = min(img_w, x1 + patch_size)
            y2 = min(img_h, y1 + patch_size)
            x1 = max(0, x2 - patch_size)
            y1 = max(0, y2 - patch_size)

            # Skip near-duplicate regions
            center_key = (y1 // 20, x1 // 20, patch_size // 20)
            if center_key in seen_centers:
                continue
            seen_centers.add(center_key)

            patch = img_rgb[y1:y2, x1:x2]
            if patch.shape[0] < 16 or patch.shape[1] < 16:
                continue

            # Re-classify this specific crop
            p_resized = cv2.resize(patch, (_INPUT_SIZE, _INPUT_SIZE))
            p_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(
                p_resized.astype(np.float32)[np.newaxis]
            )
            p_preds = model(p_tensor, training=False).numpy()
            p_decoded = tf.keras.applications.mobilenet_v2.decode_predictions(p_preds, top=1)[0]
            _, label, confidence = p_decoded[0]

            if confidence >= threshold:
                candidates.append({
                    'patch': patch,
                    'score': float(confidence),
                    'label': label,
                    'x': x1, 'y': y1,
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
        description="Extract recognizable patches from an image using GradCAM-guided search")
    parser.add_argument("image", help="Path to source image")
    parser.add_argument("--output_dir", default="image_patches",
                        help="Output folder for patches (default: image_patches)")
    parser.add_argument("--top_classes", type=int, default=10,
                        help="Number of top predictions to compute heatmaps for (default: 10)")
    parser.add_argument("--patch_scales", nargs="+", type=float, default=[0.15, 0.25, 0.4],
                        help="Patch sizes as fractions of short side (default: 0.15 0.25 0.4)")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Minimum classifier confidence to keep (default: 0.15)")
    parser.add_argument("--max_patches", type=int, default=50,
                        help="Maximum patches to save (default: 50)")

    args = parser.parse_args()

    extract_patches(
        args.image,
        output_dir=args.output_dir,
        top_classes=args.top_classes,
        patch_scales=args.patch_scales,
        threshold=args.threshold,
        max_patches=args.max_patches,
    )
