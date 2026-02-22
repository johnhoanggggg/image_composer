#!/usr/bin/env python3
"""
Auto-search: find recognizable Caltech101 images and match them against the dataset.

Pipeline:
  1. Samples candidate images from Caltech101
  2. Removes background with rembg, classifies each — keeps recognizable subjects
  3. Uses each accepted image (whole, not cropped) as a patch and searches
     a separate target pool for outline matches via template matching
  4. Visualizes all results in a summary figure

Press Play in VS Code to run with the config at the bottom.
"""

import os
import glob
import random
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_datasets as tfds

from search_caltech_rembg import (
    resize_to_max,
    get_subject_outline_neural,
    get_patch_variants,
    precompute_outlines,
    match_all_images,
    create_composite,
    _get_patch_fg_mask,
    _apply_transform,
    REMBG_AVAILABLE,
    CUDA_AVAILABLE,
)
from extract_patches import get_classifier, _INPUT_SIZE


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_image(model, img_rgb):
    """Classify an RGB image. Returns (label, confidence)."""
    resized = cv2.resize(img_rgb, (_INPUT_SIZE, _INPUT_SIZE))
    tensor = tf.keras.applications.mobilenet_v2.preprocess_input(
        resized.astype(np.float32)[np.newaxis]
    )
    preds = model(tensor, training=False).numpy()
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
    _, label, conf = decoded[0]
    return label, float(conf)


def composite_fg_on_white(img_rgb, fg_mask):
    """Composite foreground on white background for classification."""
    white = np.full_like(img_rgb, 255, dtype=np.float32)
    blended = (
        img_rgb.astype(np.float32) * fg_mask[..., None]
        + white * (1.0 - fg_mask[..., None])
    )
    return blended.astype(np.uint8)


# ============================================================
# CANDIDATE FILTERING
# ============================================================

def find_recognizable(candidates, label_names, model, threshold):
    """Remove background from each candidate, classify, keep those above threshold.

    Returns list of dicts: image, fg_mask, fg_on_white, class_name, clf_label, clf_score.
    """
    accepted = []
    for idx, example in enumerate(candidates):
        img_rgb = example["image"].numpy()
        label_idx = example["label"].numpy()
        class_name = label_names(label_idx)

        if img_rgb.shape[0] < 64 or img_rgb.shape[1] < 64:
            print(f"  [{idx+1}/{len(candidates)}] {class_name} — skipped (too small)")
            continue

        fg_mask = _get_patch_fg_mask(img_rgb)
        fg_on_white = composite_fg_on_white(img_rgb, fg_mask)
        label, conf = classify_image(model, fg_on_white)

        status = "PASS" if conf >= threshold else "skip"
        print(f"  [{idx+1}/{len(candidates)}] {class_name} -> {label} ({conf:.3f}) [{status}]")

        if conf >= threshold:
            accepted.append({
                'image': img_rgb,
                'fg_mask': fg_mask,
                'fg_on_white': fg_on_white,
                'class_name': class_name,
                'clf_label': label,
                'clf_score': conf,
            })

    return accepted


# ============================================================
# MATCHING
# ============================================================

def search_patches(accepted, targets_processed, config):
    """Run template matching for each accepted patch against the target pool."""
    for i, patch_info in enumerate(accepted):
        img_rgb = patch_info['image']
        print(
            f"\nSearching patch {i+1}/{len(accepted)}: "
            f"{patch_info['class_name']} ({patch_info['clf_label']} {patch_info['clf_score']:.2f})"
        )

        patch_proc, patch_scale = resize_to_max(img_rgb, config['max_resolution'])
        patch_outline = get_subject_outline_neural(patch_proc)

        variants = get_patch_variants(
            patch_proc, patch_outline,
            config['allow_flip'], config['allow_rotation'], config['rotation_steps']
        )
        print(f"  {len(variants)} variants")

        results = match_all_images(
            targets_processed, variants,
            config['min_scale'], config['max_scale'], config['scale_steps'],
            patch_scale, config['num_threads_matching'],
            config['early_stop_threshold'], config['max_resolution']
        )

        results.sort(key=lambda r: r['score'], reverse=True)
        patch_info['matches'] = results[:config['top_k']]
        patch_info['outline'] = patch_outline
        patch_info['patch_scale'] = patch_scale

        if patch_info['matches']:
            top = patch_info['matches'][0]
            print(f"  Best: {top['class_name']} score={top['score']:.3f}")
        else:
            print(f"  No matches found")


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_summary(accepted, blend_mode, alpha, output_path):
    """Summary figure: each row = one accepted patch, columns = original | bg-removed | top matches."""
    patches_with_matches = [p for p in accepted if p.get('matches')]
    if not patches_with_matches:
        print("No matches to visualize.")
        return

    top_k = max(len(p['matches']) for p in patches_with_matches)
    cols = 2 + top_k
    rows = len(patches_with_matches)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Patch (original)", "BG removed"] + [f"Match #{j+1}" for j in range(top_k)]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10, weight='bold')

    for r, patch_info in enumerate(patches_with_matches):
        # Col 0: original image
        axes[r, 0].imshow(patch_info['image'])
        axes[r, 0].set_ylabel(
            f"{patch_info['class_name']}\n{patch_info['clf_label']} ({patch_info['clf_score']:.2f})",
            fontsize=8, rotation=0, labelpad=80, va='center'
        )
        axes[r, 0].axis("off")

        # Col 1: bg removed (fg on white)
        axes[r, 1].imshow(patch_info['fg_on_white'])
        axes[r, 1].axis("off")

        # Cols 2+: match composites
        for j, match in enumerate(patch_info['matches']):
            col = 2 + j
            image = match['image']
            x, y, scale = match['x'], match['y'], match['scale']
            transform = match['transform']

            patch_t = _apply_transform(patch_info['image'], transform)
            fg_mask = patch_info['fg_mask']
            mask_t = _apply_transform(fg_mask, transform) if fg_mask is not None else None

            # Crop for arbitrary rotations (same logic as visualize_results)
            if mask_t is not None and 'rot' in transform:
                rot_part = transform.split('_')[0] if '_' in transform else transform
                if rot_part.startswith('rot') and rot_part not in ('rot90', 'rot180', 'rot270'):
                    mask_binary = (mask_t > 0.1 if mask_t.dtype == np.float32
                                   else mask_t > 25).astype(np.uint8) * 255
                    coords = cv2.findNonZero(mask_binary)
                    if coords is not None:
                        rx, ry, rw, rh = cv2.boundingRect(coords)
                        patch_t = patch_t[ry:ry+rh, rx:rx+rw]
                        mask_t = mask_t[ry:ry+rh, rx:rx+rw]

            composite = create_composite(
                patch_t, image, x, y, scale, blend_mode, alpha, patch_mask=mask_t
            )
            axes[r, col].imshow(composite)
            axes[r, col].set_title(
                f"{match['class_name']}\n{match['score']:.3f} ({transform})",
                fontsize=7
            )
            axes[r, col].axis("off")

        # Hide empty match columns
        for j in range(len(patch_info['matches']), top_k):
            axes[r, 2 + j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()


def visualize_debug(accepted, output_path):
    """Debug: show patch outline, best-match target outline, and red/blue overlay per patch."""
    patches_with_matches = [p for p in accepted if p.get('matches')]
    if not patches_with_matches:
        return

    n = len(patches_with_matches)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Patch outline", fontsize=10, weight='bold')
    axes[0, 1].set_title("Target outline (best)", fontsize=10, weight='bold')
    axes[0, 2].set_title("Overlay (R=patch  B=target)", fontsize=10, weight='bold')

    for r, patch_info in enumerate(patches_with_matches):
        best = patch_info['matches'][0]
        outline_var = best['outline_var']
        target_outline = best['target_outline']
        match_scale = best['match_scale']
        img_scale = best['img_scale']
        x_proc = int(best['x'] * img_scale)
        y_proc = int(best['y'] * img_scale)

        # Col 0: patch outline variant
        axes[r, 0].imshow(outline_var, cmap='gray')
        axes[r, 0].set_ylabel(
            f"{patch_info['class_name']}\n-> {best['class_name']}",
            fontsize=8, rotation=0, labelpad=80, va='center'
        )
        axes[r, 0].axis("off")

        # Col 1: target outline
        axes[r, 1].imshow(target_outline, cmap='gray')
        axes[r, 1].axis("off")

        # Col 2: overlay
        th, tw = target_outline.shape[:2]
        ph, pw = outline_var.shape[:2]
        new_w = int(pw * match_scale)
        new_h = int(ph * match_scale)

        overlay = np.zeros((th, tw, 3), dtype=np.uint8)
        overlay[:, :, 2] = target_outline  # blue = target

        if new_w > 0 and new_h > 0:
            patch_scaled = cv2.resize(outline_var, (new_w, new_h))
            x1, y1 = max(0, x_proc), max(0, y_proc)
            x2, y2 = min(tw, x_proc + new_w), min(th, y_proc + new_h)
            px1, py1 = max(0, -x_proc), max(0, -y_proc)
            px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
            if x2 > x1 and y2 > y1:
                overlay[y1:y2, x1:x2, 0] = np.maximum(
                    overlay[y1:y2, x1:x2, 0],
                    patch_scaled[py1:py2, px1:px2]
                )  # red = patch

        axes[r, 2].imshow(overlay)
        axes[r, 2].set_title(
            f"score={best['score']:.3f}  scale={match_scale:.2f}",
            fontsize=7
        )
        axes[r, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Debug outlines saved: {output_path}")
    plt.show()


# ============================================================
# CONFIGURATION — press Play in VS Code to run
# ============================================================
if __name__ == "__main__":
    OUTPUT_DIR = "auto_search_output"

    NUM_CANDIDATES = 50       # Images to screen as potential patches
    NUM_TARGETS = 200         # Separate pool to search through for matches
    CLASSIFIER_THRESHOLD = 0.3
    MAX_ACCEPTED = 10         # Keep top N most-recognizable patches
    TOP_K_MATCHES = 3         # Matches to display per patch

    # Template matching
    MIN_SCALE = 0.1
    MAX_SCALE = 0.9
    SCALE_STEPS = 20
    ALLOW_FLIP = True
    ALLOW_ROTATION = True
    ROTATION_STEPS = 8
    MAX_RESOLUTION = 320

    # Compositing
    BLEND_MODE = "replace"
    ALPHA = 1.0

    # Threading
    NUM_THREADS_SEGMENTATION = 1
    NUM_THREADS_MATCHING = 4
    EARLY_STOP_THRESHOLD = 0.95

    DEBUG_MODE = False

    # ============================================================

    if DEBUG_MODE:
        NUM_CANDIDATES = 10
        NUM_TARGETS = 20
        MAX_ACCEPTED = 3
        TOP_K_MATCHES = 2
        # Clear output folder
        if os.path.isdir(OUTPUT_DIR):
            for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
                os.remove(f)
        print("DEBUG MODE: small dataset, output folder cleared\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not REMBG_AVAILABLE:
        print("ERROR: rembg is required. Install with: uv pip install rembg")
        exit(1)

    print("Configuration:")
    print(f"  Candidates: {NUM_CANDIDATES}  |  Targets: {NUM_TARGETS}")
    print(f"  Classifier threshold: {CLASSIFIER_THRESHOLD}")
    print(f"  Max accepted patches: {MAX_ACCEPTED}")
    print(f"  Top matches per patch: {TOP_K_MATCHES}")
    print(f"  CUDA: {CUDA_AVAILABLE}")
    print()

    # ----------------------------------------------------------
    # Load dataset and split into candidates vs targets
    # ----------------------------------------------------------
    print("Loading Caltech101 dataset...")
    ds, info = tfds.load("caltech101", split="train", with_info=True, shuffle_files=True)
    ds_list = list(ds)
    label_names = info.features["label"].int2str

    all_indices = list(range(len(ds_list)))
    random.shuffle(all_indices)
    candidate_indices = all_indices[:NUM_CANDIDATES]
    target_indices = all_indices[NUM_CANDIDATES:NUM_CANDIDATES + NUM_TARGETS]

    candidates = [ds_list[i] for i in candidate_indices]
    targets = [ds_list[i] for i in target_indices]
    print(f"  {len(candidates)} candidates, {len(targets)} targets (no overlap)\n")

    # ----------------------------------------------------------
    # Step 1: classify candidates after bg removal
    # ----------------------------------------------------------
    print(f"Classifying {len(candidates)} candidates (bg removed)...")
    model = get_classifier()
    accepted = find_recognizable(candidates, label_names, model, CLASSIFIER_THRESHOLD)
    print(f"\n{len(accepted)} / {len(candidates)} passed threshold ({CLASSIFIER_THRESHOLD})")

    if not accepted:
        print("No recognizable patches found. Try lowering CLASSIFIER_THRESHOLD.")
        exit(0)

    # Keep the top MAX_ACCEPTED by classifier confidence
    accepted.sort(key=lambda p: p['clf_score'], reverse=True)
    accepted = accepted[:MAX_ACCEPTED]
    print(f"Using top {len(accepted)} patches\n")

    if DEBUG_MODE:
        for i, p in enumerate(accepted):
            fname = f"patch_{i:02d}_{p['class_name']}_{p['clf_label']}.png"
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, fname),
                cv2.cvtColor(p['fg_on_white'], cv2.COLOR_RGB2BGR)
            )
        print(f"Saved {len(accepted)} debug patches to {OUTPUT_DIR}/\n")

    # ----------------------------------------------------------
    # Step 2: precompute target outlines (once, shared across patches)
    # ----------------------------------------------------------
    print(f"Precomputing outlines for {len(targets)} targets...")
    targets_processed = precompute_outlines(
        targets, info, MAX_RESOLUTION, NUM_THREADS_SEGMENTATION
    )

    # ----------------------------------------------------------
    # Step 3: search each accepted patch
    # ----------------------------------------------------------
    config = dict(
        min_scale=MIN_SCALE, max_scale=MAX_SCALE, scale_steps=SCALE_STEPS,
        allow_flip=ALLOW_FLIP, allow_rotation=ALLOW_ROTATION,
        rotation_steps=ROTATION_STEPS, max_resolution=MAX_RESOLUTION,
        num_threads_matching=NUM_THREADS_MATCHING,
        early_stop_threshold=EARLY_STOP_THRESHOLD,
        top_k=TOP_K_MATCHES,
    )
    search_patches(accepted, targets_processed, config)

    # ----------------------------------------------------------
    # Step 4: visualize
    # ----------------------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, "auto_search_results.png")
    visualize_summary(accepted, BLEND_MODE, ALPHA, summary_path)

    if DEBUG_MODE:
        debug_path = os.path.join(OUTPUT_DIR, "auto_search_debug_outlines.png")
        visualize_debug(accepted, debug_path)

    # ----------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------
    print(f"\nSummary:")
    for i, p in enumerate(accepted):
        matches = p.get('matches', [])
        if matches:
            best = matches[0]
            print(
                f"  {i+1}. {p['class_name']} ({p['clf_label']} {p['clf_score']:.2f})"
                f" -> {best['class_name']} (score={best['score']:.3f})"
            )
        else:
            print(f"  {i+1}. {p['class_name']} ({p['clf_label']} {p['clf_score']:.2f}) -> no match")
