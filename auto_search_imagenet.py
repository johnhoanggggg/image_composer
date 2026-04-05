#!/usr/bin/env python3
"""
Auto-search with recursion: load user-chosen patches from a folder, match them
against ImageNet targets, then recursively layer more patches on top.

Pipeline:
  1. Loads patch images from PATCH_DIR, removes background
  2. Patches are matched against a target pool from ImageNet (level 0)
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
import math
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

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


def load_imagenet_targets(num_targets, split="validation"):
    """Load random targets from ImageNet-1k via HuggingFace datasets streaming."""
    print(f"Loading {num_targets} targets from ImageNet-1k ({split})...")
    ds = load_dataset(
        "imagenet-1k",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    ds_iter = iter(ds)
    first = next(ds_iter)
    print(f"  Available keys: {list(first.keys())}")

    image_key = None
    for candidate in ("image", "jpg", "png", "webp", "pixel_values"):
        if candidate in first:
            image_key = candidate
            break
    if image_key is None:
        from PIL import Image as PILImage
        for k, v in first.items():
            if isinstance(v, PILImage.Image):
                image_key = k
                break
    if image_key is None:
        raise RuntimeError(f"Cannot find image key in dataset. Keys: {list(first.keys())}")

    label_key = None
    for candidate in ("label", "cls", "class", "target", "fine_label"):
        if candidate in first:
            label_key = candidate
            break
    if label_key is None:
        raise RuntimeError(f"Cannot find label key in dataset. Keys: {list(first.keys())}")

    print(f"  Using image_key='{image_key}', label_key='{label_key}'")

    label_names = None
    try:
        features = load_dataset(
            "imagenet-1k",
            split=split,
            streaming=True,
            trust_remote_code=True,
        ).features
        label_feature = features[label_key]
        label_names = label_feature.names  
    except Exception:
        pass

    def extract(item):
        img = item[image_key]
        if img.mode != "RGB":
            img = img.convert("RGB")
        lbl = item[label_key]
        class_name = label_names[lbl] if label_names else f"class_{lbl}"
        return {
            "image": np.array(img),
            "label": lbl,
            "class_name": class_name,
        }

    samples = [extract(first)]
    for i, item in enumerate(ds_iter):
        idx = i + 1
        if len(samples) < num_targets:
            samples.append(extract(item))
        else:
            j = random.randint(0, idx)
            if j < num_targets:
                samples[j] = extract(item)
        if idx >= num_targets * 10:
            break

    print(f"  {len(samples)} targets loaded")
    return samples


def precompute_outlines_imagenet(targets, max_resolution, num_threads):
    """Compute subject outlines for ImageNet targets."""
    import concurrent.futures

    def process_one(t):
        img_rgb = t["image"]
        label = t["class_name"]
        img_proc, img_scale = resize_to_max(img_rgb, max_resolution)
        outline = get_subject_outline_neural(img_proc)
        return {
            "image": img_rgb,           # Retain Full Resolution Image
            "outline": outline,
            "img_scale": img_scale,
            "class_name": label,
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = {ex.submit(process_one, t): i for i, t in enumerate(targets)}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                results.append(future.result())
                print(f"  Outline {i+1}/{len(targets)}", end="\r")
            except Exception as e:
                print(f"  Outline failed: {e}")

    print()
    return results


def load_patches_from_folder(patch_dir):
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
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if img_rgb.shape[0] < 64 or img_rgb.shape[1] < 64:
            continue

        fg_mask = _get_patch_fg_mask(img_rgb)
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
    """Matcher inherently returns x, y, and scale scaled for the Full Resolution image."""
    transform = match_result['transform']
    x, y, scale = match_result['x'], match_result['y'], match_result['scale']

    patch_t = _apply_transform(patch_img, transform)
    mask_t = _apply_transform(fg_mask, transform) if fg_mask is not None else None

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
            'history': [best['image'].copy(), composite.copy()],
        })
    return canvases


def prepare_patch_data(accepted_patches, config):
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
    if not new_accepted:
        return

    print(f"  Precomputing outlines for {len(new_accepted)} new patches...")
    patch_data = prepare_patch_data(new_accepted, config)

    print(f"  Computing outlines for {len(canvases)} canvases...")
    canvas_processed = [
        outline_from_image(c['image'], c['label'], config['max_resolution'])
        for c in canvases
    ]

    for c_idx, canvas in enumerate(canvases):
        target = [canvas_processed[c_idx]]
        best_score = -1
        best_result = None
        best_patch_info = None

        for pd in patch_data:
            results = match_all_images(
                target, pd['variants'],
                config['min_scale'], config['max_scale'], config['scale_steps'],
                pd['patch_scale'], 1,
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
    """
    Creates one window per patch. Lays out exactly the top 25 matches 
    in a perfect 5x5 square grid.
    """
    patches_with_matches = [p for p in accepted if p.get('matches')]
    if not patches_with_matches:
        print("No matches to visualize.")
        return

    base, ext = os.path.splitext(output_path)
    save_dir = os.path.dirname(output_path) or "."
    saved_count = [0]

    for patch_idx, patch_info in enumerate(patches_with_matches):
        matches = patch_info['matches']
        # Cap at 25 to ensure it fits perfectly in a 5x5 grid
        matches = matches[:25] 
        
        # Perfect 5x5 grid
        rows, cols = 5, 5
        
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.atleast_1d(axes).flatten()
        
        cell_map = {}
        
        # Populate all 25 cells strictly with matches
        for j, match in enumerate(matches):
            if j >= len(axes):
                break
            ax = axes[j]
            
            composite = make_composite(
                patch_info['image'], patch_info['fg_mask'],
                match['image'], match, blend_mode, alpha
            )
            
            ax.imshow(composite)
            ax.set_title(
                f"#{j+1}: {match['class_name']}\nscore={match['score']:.3f} s={match['scale']:.2f}",
                fontsize=8
            )
            ax.axis("off")
            cell_map[ax] = (composite, patch_info['class_name'], match['class_name'])
            
        # Hide any unused cells (in case there are fewer than 25 matches found)
        for j in range(len(matches), len(axes)):
            axes[j].axis("off")
            
        def on_click(event, cmap=cell_map):
            if event.inaxes is None:
                return
            if event.inaxes in cmap:
                img, patch_name, match_name = cmap[event.inaxes]
                saved_count[0] += 1
                fname = f"saved_{patch_name}+{match_name}_{saved_count[0]:03d}.png"
                fpath = os.path.join(save_dir, fname)
                cv2.imwrite(fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"Saved: {fpath}")
                
                event.inaxes.axis("on")
                event.inaxes.set_xticks([])
                event.inaxes.set_yticks([])
                for spine in event.inaxes.spines.values():
                    spine.set_edgecolor('lime')
                    spine.set_linewidth(4)
                event.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.suptitle(f"Top 25 Matches for: {patch_info['class_name']} (Click match to save)", fontsize=12, weight='bold')
        plt.tight_layout()
        
        page_path = f"{base}_{patch_info['class_name']}{ext}"
        plt.savefig(page_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {page_path}")
        
        plt.show()


def visualize_evolution(canvases, output_path):
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
        
        # Scale coordinates down relative to the downscaled outline view
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
        overlay[:, :, 2] = target_outline

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
    PATCH_DIR = "C:\\Users\\johnh\\Documents\\image_composer\\image_composer\\image_patches"
    OUTPUT_DIR = "auto_search_output"

    # INCREASED NUM TARGETS to ensure we can actually find 25 decent candidates
    NUM_TARGETS = 200 

    MATCH_THRESHOLD = 0.2
    # TOP 25 CANDIDATES
    TOP_K_MATCHES = 25

    IMAGENET_SPLIT = "train"   

    RECURSION_DEPTH = 0

    MIN_SCALE = 0.3
    MAX_SCALE = 0.8
    SCALE_STEPS = 20
    ALLOW_FLIP = True
    ALLOW_ROTATION = True
    ROTATION_STEPS = 16
    MAX_RESOLUTION = 320

    BLEND_MODE = "replace"
    ALPHA = 1.0

    NUM_THREADS_SEGMENTATION = 1
    NUM_THREADS_MATCHING = 8
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
        print(f"ERROR: patch folder '{PATCH_DIR}' not found.")
        exit(1)

    print("Configuration:")
    print(f"  Patch folder:    {PATCH_DIR}")
    print(f"  Targets:         {NUM_TARGETS}")
    print(f"  ImageNet split:  {IMAGENET_SPLIT}")
    print(f"  Match threshold: {MATCH_THRESHOLD}")
    print(f"  Recursion depth: {RECURSION_DEPTH}")
    print(f"  CUDA:            {CUDA_AVAILABLE}")
    print()

    # ----------------------------------------------------------
    # Step 1: load patches
    # ----------------------------------------------------------
    print(f"Loading patches from {PATCH_DIR}/...")
    accepted = load_patches_from_folder(PATCH_DIR)
    print(f"\n{len(accepted)} patches loaded\n")

    if not accepted:
        print(f"No valid images in {PATCH_DIR}/. Add .png/.jpg files and retry.")
        exit(0)

    # ----------------------------------------------------------
    # Step 2: load ImageNet targets
    # ----------------------------------------------------------
    targets = load_imagenet_targets(NUM_TARGETS, split=IMAGENET_SPLIT)

    # ----------------------------------------------------------
    # Step 3: precompute target outlines
    # ----------------------------------------------------------
    print(f"Precomputing outlines for {len(targets)} targets...")
    targets_processed = precompute_outlines_imagenet(
        targets, MAX_RESOLUTION, NUM_THREADS_SEGMENTATION
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
    # Step 4: match patches against targets
    # ----------------------------------------------------------
    search_patches(accepted, targets_processed, config)

    before = len(accepted)
    accepted = [p for p in accepted if p.get('matches')]
    if before != len(accepted):
        print(f"\n{before - len(accepted)} patches dropped (no match >= {MATCH_THRESHOLD})")
    print(f"{len(accepted)} patches with matches\n")

    if not accepted:
        print("No patches matched above threshold. Lower MATCH_THRESHOLD or add more targets.")
        exit(0)

    # ----------------------------------------------------------
    # Step 5: visualize level-0
    # ----------------------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, "level0_results.png")
    visualize_summary(accepted, BLEND_MODE, ALPHA, summary_path)

    if DEBUG_MODE:
        debug_path = os.path.join(OUTPUT_DIR, "level0_debug_outlines.png")
        visualize_debug(accepted, debug_path)

    # ----------------------------------------------------------
    # Step 6: build canvases and recurse
    # ----------------------------------------------------------
    canvases = None
    if RECURSION_DEPTH > 0:
        canvases = build_canvases(accepted, BLEND_MODE, ALPHA)
        print(f"\nBuilt {len(canvases)} canvases for recursion\n")

        for level in range(RECURSION_DEPTH):
            print(f"{'='*60}")
            print(f"RECURSION LEVEL {level+1}/{RECURSION_DEPTH}")
            print(f"{'='*60}")
            recurse_level(canvases, accepted, config, BLEND_MODE, ALPHA)

        evo_path = os.path.join(OUTPUT_DIR, "evolution.png")
        visualize_evolution(canvases, evo_path)

        for i, canvas in enumerate(canvases):
            final_path = os.path.join(OUTPUT_DIR, f"chaos_{i:02d}.png")
            cv2.imwrite(final_path,
                        cv2.cvtColor(canvas['image'], cv2.COLOR_RGB2BGR))
        print(f"\nSaved {len(canvases)} final chaotic images to {OUTPUT_DIR}/")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\nFinal Summary:")
    for i, p in enumerate(accepted):
        matches = p.get('matches', [])
        if matches:
            best = matches[0]
            print(f"  {i+1}. {p['class_name']} -> {best['class_name']} "
                  f"(score={best['score']:.3f})")
        else:
            print(f"  {i+1}. {p['class_name']} -> no match")

    if canvases:
        print(f"\nCanvases after {RECURSION_DEPTH} recursion levels:")
        for i, c in enumerate(canvases):
            print(f"  {i}. {c['label']}  ({len(c['history'])} snapshots)")