#!/usr/bin/env python3
"""
Extract recognizable patches from Caltech101 images using GradCAM-guided extraction.

Instead of sliding a window and classifying every crop (slow), this:
  1. Loads N random images from the Caltech101 dataset
  2. Runs the classifier ONCE on each full image to get top predictions
  3. Computes GradCAM heatmaps to locate where each prediction fires
  4. Uses the heatmap activation extent to adaptively size each patch
     (instead of arbitrary fixed scales that can over-crop subjects)
  5. Re-classifies each candidate patch to confirm + get final scores

Goal: build a folder of recognizable patches to later match against
larger target images with search_caltech_rembg.py.

Press Play in VS Code to run with the config below.
"""

import os
import math
import random
import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_datasets as tfds

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
    last_conv = model.get_layer("out_relu")

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(img_tensor, training=False)
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(conv_out * weights, axis=-1).numpy()[0]

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


def activation_bbox(cam, peak_y, peak_x, activation_thresh=0.3, padding=0.25,
                    min_frac=0.15):
    """Compute a bounding box around the activation region near a peak.

    Instead of using arbitrary fixed scales, this measures the actual extent
    of the GradCAM activation around a peak and adds padding.

    Returns (cy_frac, cx_frac, h_frac, w_frac) as fractions of heatmap size.
    """
    cam_h, cam_w = cam.shape
    mask = (cam >= activation_thresh * cam.max()).astype(np.uint8)

    # Label connected components, find the one containing the peak
    num_labels, labels = cv2.connectedComponents(mask)
    peak_label = labels[peak_y, peak_x]

    if peak_label == 0:
        # Peak not in any activation region; use a default size
        return peak_y / cam_h, peak_x / cam_w, min_frac, min_frac

    region = (labels == peak_label).astype(np.uint8)
    ys, xs = np.where(region)
    ry1, ry2 = ys.min(), ys.max()
    rx1, rx2 = xs.min(), xs.max()

    rh = (ry2 - ry1 + 1) / cam_h
    rw = (rx2 - rx1 + 1) / cam_w
    cy = (ry1 + ry2) / 2.0 / cam_h
    cx = (rx1 + rx2) / 2.0 / cam_w

    # Add padding so the patch isn't tight-cropped
    rh = rh * (1 + padding)
    rw = rw * (1 + padding)

    # Enforce minimum size so tiny activations still produce usable patches
    rh = max(rh, min_frac)
    rw = max(rw, min_frac)

    return cy, cx, rh, rw


def extract_patches_from_image(img_rgb, image_name, output_dir, model,
                               top_classes=10, threshold=0.15,
                               max_patches=10, padding=0.25, min_frac=0.15):
    """Extract recognizable patches from a single image using adaptive GradCAM sizing.

    Args:
        img_rgb: RGB numpy array.
        image_name: Base name for saved files.
        output_dir: Folder to save accepted patches into.
        model: Pre-loaded classifier model.
        top_classes: Number of top classifier predictions to generate heatmaps for.
        threshold: Minimum classifier confidence to keep a re-classified patch.
        max_patches: Maximum number of patches to save per image.
        padding: Fractional padding around the GradCAM activation region (0.25 = 25%).
        min_frac: Minimum patch size as fraction of image dimension.
    """
    img_h, img_w = img_rgb.shape[:2]

    # Step 1: classify full image once
    resized = cv2.resize(img_rgb, (_INPUT_SIZE, _INPUT_SIZE))
    img_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(
        resized.astype(np.float32)[np.newaxis]
    )
    preds = model(img_tensor, training=False).numpy()
    top_indices = np.argsort(preds[0])[::-1][:top_classes]

    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=top_classes)[0]
    print(f"  Top predictions: {', '.join(f'{l}({c:.2f})' for _, l, c in decoded[:5])}")

    # Step 2: compute GradCAM for each top class, get adaptive bounding boxes
    candidate_regions = []

    for rank, class_idx in enumerate(top_indices):
        cam = gradcam(model, img_tensor, int(class_idx))
        peaks = find_peaks(cam, min_distance=3, threshold_rel=0.3)

        _, label, _ = decoded[rank]

        for py, px, intensity in peaks[:3]:
            cy, cx, rh, rw = activation_bbox(
                cam, py, px, activation_thresh=0.3,
                padding=padding, min_frac=min_frac
            )
            candidate_regions.append((cy, cx, rh, rw, int(class_idx), label, intensity))

    # Step 3: extract patches using adaptive bounding boxes
    candidates = []
    seen_centers = set()

    for cy_frac, cx_frac, h_frac, w_frac, class_idx, cam_label, intensity in candidate_regions:
        patch_h = max(32, int(h_frac * img_h))
        patch_w = max(32, int(w_frac * img_w))
        cy_px = int(cy_frac * img_h)
        cx_px = int(cx_frac * img_w)

        y1 = max(0, cy_px - patch_h // 2)
        x1 = max(0, cx_px - patch_w // 2)
        y2 = min(img_h, y1 + patch_h)
        x2 = min(img_w, x1 + patch_w)
        y1 = max(0, y2 - patch_h)
        x1 = max(0, x2 - patch_w)

        # Skip near-duplicate regions
        center_key = (y1 // 20, x1 // 20, patch_h // 20, patch_w // 20)
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
                'w': x2 - x1, 'h': y2 - y1,
            })

    # Sort by score descending
    candidates.sort(key=lambda c: c['score'], reverse=True)

    # Deduplicate overlapping patches
    kept = []
    for c in candidates:
        cx = c['x'] + c['w'] // 2
        cy = c['y'] + c['h'] // 2
        overlap = False
        for k in kept:
            if (k['x'] <= cx < k['x'] + k['w'] and
                    k['y'] <= cy < k['y'] + k['h']):
                overlap = True
                break
        if not overlap:
            kept.append(c)
        if len(kept) >= max_patches:
            break

    # Save patches
    saved_paths = []
    for i, c in enumerate(kept):
        safe_label = c['label'].replace(' ', '_')
        fname = f"{image_name}_p{i:03d}_{safe_label}_{c['score']:.2f}.png"
        out_path = os.path.join(output_dir, fname)
        patch_bgr = cv2.cvtColor(c['patch'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, patch_bgr)
        saved_paths.append(out_path)

    return kept, saved_paths


def extract_from_caltech(num_images=10, output_dir="image_patches",
                         top_classes=10, threshold=0.15, max_patches_per_image=5,
                         padding=0.25, min_frac=0.15):
    """Extract patches from N random Caltech101 images.

    Args:
        num_images: Number of random images to sample from the dataset.
        output_dir: Folder to save accepted patches into.
        top_classes: Number of top classifier predictions to generate heatmaps for.
        threshold: Minimum classifier confidence to keep a re-classified patch.
        max_patches_per_image: Maximum number of patches to save per source image.
        padding: Fractional padding around GradCAM activation (0.25 = 25%).
        min_frac: Minimum patch size as fraction of image dimension.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = get_classifier()

    print("Loading Caltech101 dataset...")
    ds, info = tfds.load("caltech101", split="train", with_info=True, shuffle_files=True)
    ds_list = list(ds)

    if num_images < len(ds_list):
        samples = random.sample(ds_list, num_images)
    else:
        samples = ds_list

    label_names = info.features["label"].int2str

    all_kept = []
    all_paths = []

    for idx, example in enumerate(samples):
        img_rgb = example["image"].numpy()
        label_idx = example["label"].numpy()
        class_name = label_names(label_idx)
        image_name = f"cal_{class_name}_{idx:04d}"

        # Skip very small images
        if img_rgb.shape[0] < 64 or img_rgb.shape[1] < 64:
            print(f"[{idx+1}/{len(samples)}] {class_name} — skipped (too small)")
            continue

        print(f"[{idx+1}/{len(samples)}] {class_name} ({img_rgb.shape[1]}x{img_rgb.shape[0]})")

        kept, paths = extract_patches_from_image(
            img_rgb, image_name, output_dir, model,
            top_classes=top_classes,
            threshold=threshold,
            max_patches=max_patches_per_image,
            padding=padding,
            min_frac=min_frac,
        )

        print(f"  -> {len(kept)} patches saved")
        all_kept.extend(kept)
        all_paths.extend(paths)

    print(f"\nDone. {len(all_kept)} total patches from {len(samples)} images.")
    print(f"Saved to: {output_dir}/")

    if all_kept:
        _visualize_grid(all_kept, output_dir)

    return all_kept, all_paths


def _visualize_grid(kept, output_dir):
    """Save a grid of all extracted patches."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(kept)
    cols = min(8, math.ceil(math.sqrt(n)))
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
    grid_path = os.path.join(output_dir, "caltech_patches_grid.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grid saved: {grid_path}")


# ============================================================
# CONFIGURATION — press Play in VS Code to run
# ============================================================
if __name__ == "__main__":
    OUTPUT_DIR = "image_patches"
    NUM_IMAGES = 10          # Number of random Caltech101 images to extract from
    TOP_CLASSES = 10         # GradCAM heatmaps per image
    THRESHOLD = 0.15         # Min confidence to keep a re-classified patch
    MAX_PATCHES_PER_IMAGE = 5

    # Adaptive sizing controls (replaces fixed patch_scales)
    PADDING = 0.25           # 25% padding around the GradCAM activation region
    MIN_FRAC = 0.15          # Minimum patch size as fraction of image dimension

    print(f"Configuration:")
    print(f"  Source: {NUM_IMAGES} random Caltech101 images")
    print(f"  Top classes per image: {TOP_CLASSES}")
    print(f"  Confidence threshold: {THRESHOLD}")
    print(f"  Max patches per image: {MAX_PATCHES_PER_IMAGE}")
    print(f"  GradCAM padding: {PADDING*100:.0f}%")
    print(f"  Min patch fraction: {MIN_FRAC}")
    print()

    extract_from_caltech(
        num_images=NUM_IMAGES,
        output_dir=OUTPUT_DIR,
        top_classes=TOP_CLASSES,
        threshold=THRESHOLD,
        max_patches_per_image=MAX_PATCHES_PER_IMAGE,
        padding=PADDING,
        min_frac=MIN_FRAC,
    )
