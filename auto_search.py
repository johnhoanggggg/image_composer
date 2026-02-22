#!/usr/bin/env python3
"""
Auto-search with recursion: load user-chosen patches from a folder, match them
against Caltech101 targets, then recursively layer more patches on top.

Pipeline:
  1. Loads patch images from PATCH_DIR, removes background
  2. Patches are matched against a target pool from Caltech101 (level 0)
  3. Creates initial composite canvases from best matches
  4. For each recursion level:
     - Cycles through patches again
     - Matches every patch against each canvas, picks the best per canvas
     - Layers the winning patch on top -> canvas gets more chaotic
  5. Visualizes level-0 results + evolution strip per canvas

Press Play in VS Code to run with the config at the bottom.
"""

import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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

# ============================================================
# HELPERS
# ============================================================

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


def load_patches_from_folder(patch_dir):
    """Load all images from patch_dir, remove background, return accepted list.

    Returns list of dicts with: image, fg_mask, fg_on_white, class_name.
    """
    files = sorted(
        f for f in os.listdir(patch_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    if not files:
        print(f"No image files found in {patch_dir}")
        return []

    accepted = []
    for i, fname in enumerate(files):
        path = os.path.join(patch_dir, fname)
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  [{i+1}/{len(files)}] {fname} — skipped (unreadable)")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if img_rgb.shape[0] < 64 or img_rgb.shape[1] < 64:
            print(f"  [{i+1}/{len(files)}] {fname} — skipped (too small)")
            continue

        fg_mask = _get_patch_fg_mask(img_rgb)
        # Composite on white for visualization
        white = np.full_like(img_rgb, 255, dtype=np.float32)
        fg_on_white = (
            img_rgb.astype(np.float32) * fg_mask[..., None]
            + white * (1.0 - fg_mask[..., None])
        ).astype(np.uint8)

        class_name = os.path.splitext(fname)[0]
        print(f"  [{i+1}/{len(files)}] {fname} — loaded")

        accepted.append({
            'image': img_rgb,
            'fg_mask': fg_mask,
            'fg_on_white': fg_on_white,
            'class_name': class_name,
            'clf_label': class_name,
            'clf_score': 1.0,
        })

    return accepted


def make_composite(patch_img, fg_mask, target_img, match_result, blend_mode, alpha):
    """Composite a patch onto a target, handling transforms and arbitrary rotation crop."""
    transform = match_result['transform']
    x, y, scale = match_result['x'], match_result['y'], match_result['scale']

    patch_t = _apply_transform(patch_img, transform)
    mask_t = _apply_transform(fg_mask, transform) if fg_mask is not None else None

    # Crop tight bbox for arbitrary (non-cardinal) rotations
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

    return create_composite(patch_t, target_img, x, y, scale,
                            blend_mode, alpha, patch_mask=mask_t)


def outline_from_image(img_rgb, label, max_resolution):
    """Compute an outline dict from a raw image (same shape as precompute_outlines returns)."""
    img_proc, img_scale = resize_to_max(img_rgb, max_resolution)
    outline = get_subject_outline_neural(img_proc)
    return {
        'image': img_rgb,
        'outline': outline,
        'img_scale': img_scale,
        'class_name': label,
    }


# ============================================================
# MATCHING
# ============================================================

def search_patches(accepted, targets_processed, config):
    """Run template matching for each accepted patch against the target pool."""
    for i, patch_info in enumerate(accepted):
        img_rgb = patch_info['image']
        print(f"\nSearching patch {i+1}/{len(accepted)}: {patch_info['class_name']}")

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
        match_thresh = config.get('match_threshold', 0.0)
        results = [r for r in results if r['score'] >= match_thresh]
        patch_info['matches'] = results[:config['top_k']]
        patch_info['outline'] = patch_outline
        patch_info['patch_scale'] = patch_scale

        if patch_info['matches']:
            top = patch_info['matches'][0]
            print(f"  Best: {top['class_name']} score={top['score']:.3f}")
        else:
            print(f"  No matches above {match_thresh}")


# ============================================================
# RECURSION
# ============================================================

def build_canvases(accepted, blend_mode, alpha):
    """Create initial composite canvases from level-0 best matches."""
    canvases = []
    for patch_info in accepted:
        if not patch_info.get('matches'):
            continue
        best = patch_info['matches'][0]
        composite = make_composite(
            patch_info['image'], patch_info['fg_mask'],
            best['image'], best, blend_mode, alpha
        )
        canvases.append({
            'image': composite,
            'label': f"{patch_info['class_name']}+{best['class_name']}",
            # history[0] = bare target, history[1] = after level 0
            'history': [best['image'].copy(), composite.copy()],
        })
    return canvases


def prepare_patch_data(accepted_patches, config):
    """Precompute outlines and variants for accepted patches (done once, reused per canvas)."""
    patch_data = []
    for p in accepted_patches:
        patch_proc, patch_scale = resize_to_max(p['image'], config['max_resolution'])
        outline = get_subject_outline_neural(patch_proc)
        variants = get_patch_variants(
            patch_proc, outline,
            config['allow_flip'], config['allow_rotation'], config['rotation_steps']
        )
        patch_data.append({
            'patch_info': p,
            'variants': variants,
            'patch_scale': patch_scale,
        })
    return patch_data


def recurse_level(canvases, new_accepted, config, blend_mode, alpha):
    """One recursion level: match new patches against canvases, composite the best onto each."""
    if not new_accepted:
        return

    # Precompute outlines + variants for new patches once
    print(f"  Precomputing outlines for {len(new_accepted)} new patches...")
    patch_data = prepare_patch_data(new_accepted, config)

    # Compute outlines for current canvases
    print(f"  Computing outlines for {len(canvases)} canvases...")
    canvas_processed = [
        outline_from_image(c['image'], c['label'], config['max_resolution'])
        for c in canvases
    ]

    # For each canvas, try every new patch and keep the best
    for c_idx, canvas in enumerate(canvases):
        target = [canvas_processed[c_idx]]  # single-element list for match_all_images
        best_score = -1
        best_result = None
        best_patch_info = None

        for pd in patch_data:
            results = match_all_images(
                target, pd['variants'],
                config['min_scale'], config['max_scale'], config['scale_steps'],
                pd['patch_scale'], 1,  # 1 thread — single target
                config['early_stop_threshold'], config['max_resolution']
            )
            if results and results[0]['score'] > best_score:
                best_score = results[0]['score']
                best_result = results[0]
                best_patch_info = pd['patch_info']

        match_thresh = config.get('match_threshold', 0.0)
        if best_result and best_patch_info and best_score >= match_thresh:
            composite = make_composite(
                best_patch_info['image'], best_patch_info['fg_mask'],
                canvas['image'], best_result, blend_mode, alpha
            )
            canvas['image'] = composite
            canvas['label'] += f"+{best_patch_info['class_name']}"
            canvas['history'].append(composite.copy())
            print(f"    Canvas {c_idx}: +{best_patch_info['class_name']} (score={best_score:.3f})")
        else:
            print(f"    Canvas {c_idx}: no match above {match_thresh} (best={best_score:.3f})")


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_summary(accepted, blend_mode, alpha, output_path):
    """Level-0 summary: each row = one patch, columns = original | bg-removed | matches."""
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
        axes[r, 0].imshow(patch_info['image'])
        axes[r, 0].set_ylabel(
            patch_info['class_name'],
            fontsize=8, rotation=0, labelpad=80, va='center'
        )
        axes[r, 0].axis("off")

        axes[r, 1].imshow(patch_info['fg_on_white'])
        axes[r, 1].axis("off")

        for j, match in enumerate(patch_info['matches']):
            col = 2 + j
            composite = make_composite(
                patch_info['image'], patch_info['fg_mask'],
                match['image'], match, blend_mode, alpha
            )
            axes[r, col].imshow(composite)
            axes[r, col].set_title(
                f"{match['class_name']}\n{match['score']:.3f} ({match['transform']})",
                fontsize=7
            )
            axes[r, col].axis("off")

        for j in range(len(patch_info['matches']), top_k):
            axes[r, 2 + j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()


def visualize_evolution(canvases, output_path):
    """Show each canvas progressing through recursion levels (one row per canvas)."""
    if not canvases:
        return

    max_levels = max(len(c['history']) for c in canvases)
    rows = len(canvases)
    cols = max_levels

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    for c in range(cols):
        label = "Original target" if c == 0 else f"+ Level {c}"
        axes[0, c].set_title(label, fontsize=10, weight='bold')

    for r, canvas in enumerate(canvases):
        short_label = canvas['label'].split('+')[0]
        axes[r, 0].set_ylabel(short_label, fontsize=8, rotation=0, labelpad=60, va='center')
        for c, snapshot in enumerate(canvas['history']):
            axes[r, c].imshow(snapshot)
            axes[r, c].axis("off")
        for c in range(len(canvas['history']), cols):
            axes[r, c].axis("off")

    plt.suptitle("Recursive Compositing Evolution", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Evolution saved: {output_path}")
    plt.show()


def visualize_debug(accepted, output_path):
    """Debug: outline overlay for level-0 matches."""
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

        axes[r, 0].imshow(outline_var, cmap='gray')
        axes[r, 0].set_ylabel(
            f"{patch_info['class_name']}\n-> {best['class_name']}",
            fontsize=8, rotation=0, labelpad=80, va='center'
        )
        axes[r, 0].axis("off")

        axes[r, 1].imshow(target_outline, cmap='gray')
        axes[r, 1].axis("off")

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
                )

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
    PATCH_DIR = "patches"             # Folder of user-chosen patch images
    OUTPUT_DIR = "auto_search_output"

    NUM_TARGETS = 200         # Caltech101 images to search for matches
    MATCH_THRESHOLD = 0.4     # Min template match score to accept a match
    TOP_K_MATCHES = 3         # Matches to display per patch (level 0)

    # Recursion — layer additional patches onto canvases
    RECURSION_DEPTH = 2               # 0 = no recursion, N = N extra layers per canvas

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
        NUM_TARGETS = 20
        TOP_K_MATCHES = 2
        RECURSION_DEPTH = 1
        if os.path.isdir(OUTPUT_DIR):
            for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
                os.remove(f)
        print("DEBUG MODE: small dataset, output folder cleared\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not REMBG_AVAILABLE:
        print("ERROR: rembg is required. Install with: uv pip install rembg")
        exit(1)

    if not os.path.isdir(PATCH_DIR):
        print(f"ERROR: patch folder '{PATCH_DIR}' not found. Create it and add images.")
        exit(1)

    print("Configuration:")
    print(f"  Patch folder: {PATCH_DIR}")
    print(f"  Targets: {NUM_TARGETS}")
    print(f"  Match threshold: {MATCH_THRESHOLD}")
    print(f"  Recursion depth: {RECURSION_DEPTH}")
    print(f"  CUDA: {CUDA_AVAILABLE}")
    print()

    # ----------------------------------------------------------
    # Step 1: load patches from folder
    # ----------------------------------------------------------
    print(f"Loading patches from {PATCH_DIR}/...")
    accepted = load_patches_from_folder(PATCH_DIR)
    print(f"\n{len(accepted)} patches loaded\n")

    if not accepted:
        print(f"No valid images in {PATCH_DIR}/. Add .png/.jpg files and retry.")
        exit(0)

    # ----------------------------------------------------------
    # Step 2: load Caltech101 target pool
    # ----------------------------------------------------------
    print("Loading Caltech101 dataset...")
    ds, info = tfds.load("caltech101", split="train", with_info=True, shuffle_files=True)
    ds_list = list(ds)

    all_indices = list(range(len(ds_list)))
    random.shuffle(all_indices)
    target_indices = all_indices[:NUM_TARGETS]

    targets = [ds_list[i] for i in target_indices]
    print(f"  {len(targets)} targets reserved\n")

    # ----------------------------------------------------------
    # Step 3: precompute target outlines
    # ----------------------------------------------------------
    print(f"Precomputing outlines for {len(targets)} targets...")
    targets_processed = precompute_outlines(
        targets, info, MAX_RESOLUTION, NUM_THREADS_SEGMENTATION
    )

    config = dict(
        min_scale=MIN_SCALE, max_scale=MAX_SCALE, scale_steps=SCALE_STEPS,
        allow_flip=ALLOW_FLIP, allow_rotation=ALLOW_ROTATION,
        rotation_steps=ROTATION_STEPS, max_resolution=MAX_RESOLUTION,
        num_threads_matching=NUM_THREADS_MATCHING,
        early_stop_threshold=EARLY_STOP_THRESHOLD,
        top_k=TOP_K_MATCHES,
        match_threshold=MATCH_THRESHOLD,
    )

    # ----------------------------------------------------------
    # Step 4: match all patches against targets, retry skipped ones
    # ----------------------------------------------------------
    search_patches(accepted, targets_processed, config)

    # Drop patches that got no matches above threshold
    before = len(accepted)
    accepted = [p for p in accepted if p.get('matches')]
    if before != len(accepted):
        print(f"\n{before - len(accepted)} patches dropped (no match >= {MATCH_THRESHOLD})")
    print(f"{len(accepted)} patches with matches\n")

    if not accepted:
        print("No patches matched above threshold. Lower MATCH_THRESHOLD or add more targets.")
        exit(0)

    # Level-0 visualization
    summary_path = os.path.join(OUTPUT_DIR, "level0_results.png")
    visualize_summary(accepted, BLEND_MODE, ALPHA, summary_path)

    if DEBUG_MODE:
        debug_path = os.path.join(OUTPUT_DIR, "level0_debug_outlines.png")
        visualize_debug(accepted, debug_path)

    # ----------------------------------------------------------
    # Step 5: build canvases and recurse
    # ----------------------------------------------------------
    canvases = None
    if RECURSION_DEPTH > 0:
        canvases = build_canvases(accepted, BLEND_MODE, ALPHA)
        print(f"\nBuilt {len(canvases)} canvases for recursion\n")

        for level in range(RECURSION_DEPTH):
            print(f"{'='*60}")
            print(f"RECURSION LEVEL {level+1}/{RECURSION_DEPTH}")
            print(f"{'='*60}")

            # Reuse the same user-chosen patches for each recursion level
            recurse_level(canvases, accepted, config, BLEND_MODE, ALPHA)

        # Evolution visualization
        evo_path = os.path.join(OUTPUT_DIR, "evolution.png")
        visualize_evolution(canvases, evo_path)

        # Save final chaotic images
        for i, canvas in enumerate(canvases):
            final_path = os.path.join(OUTPUT_DIR, f"chaos_{i:02d}.png")
            cv2.imwrite(final_path, cv2.cvtColor(canvas['image'], cv2.COLOR_RGB2BGR))
        print(f"\nSaved {len(canvases)} final chaotic images to {OUTPUT_DIR}/")

    # ----------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------
    print(f"\nFinal Summary:")
    for i, p in enumerate(accepted):
        matches = p.get('matches', [])
        if matches:
            best = matches[0]
            print(
                f"  {i+1}. {p['class_name']}"
                f" -> {best['class_name']} (score={best['score']:.3f})"
            )
        else:
            print(f"  {i+1}. {p['class_name']} -> no match")

    if canvases:
        print(f"\nCanvases after {RECURSION_DEPTH} recursion levels:")
        for i, c in enumerate(canvases):
            print(f"  {i}. {c['label']}  ({len(c['history'])} snapshots)")
