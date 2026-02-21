#!/usr/bin/env python3
"""
Search Caltech101 dataset for best matching images to composite a patch onto.
Uses outline matching from composite.py approach.
"""

import os
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


def get_subject_outline(img, margin=5, dilate=1, force_outer=True):
    """
    Extract subject outline using GrabCut.
    """
    if len(img.shape) == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    corner_size = max(5, min(h, w) // 20)
    corners = [
        gray[0:corner_size, 0:corner_size],
        gray[0:corner_size, w-corner_size:w],
        gray[h-corner_size:h, 0:corner_size],
        gray[h-corner_size:h, w-corner_size:w]
    ]
    corner_stds = [np.std(c) for c in corners]
    avg_corner_std = np.mean(corner_stds)
    
    if avg_corner_std < 20:
        bg_color = np.median([np.median(c) for c in corners])
        diff = np.abs(gray.astype(np.float32) - bg_color)
        fg_mask = (diff > 25).astype(np.uint8) * 255
    else:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = cv2.GC_PR_BGD
        
        cx, cy = w // 2, h // 2
        rect_w, rect_h = int(w * 0.7), int(h * 0.7)
        x1, y1 = max(0, cx - rect_w // 2), max(0, cy - rect_h // 2)
        x2, y2 = min(w, cx + rect_w // 2), min(h, cy + rect_h // 2)
        mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
        
        mask[:margin, :] = cv2.GC_BGD
        mask[-margin:, :] = cv2.GC_BGD
        mask[:, :margin] = cv2.GC_BGD
        mask[:, -margin:] = cv2.GC_BGD
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(img_bgr, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except:
            return cv2.Canny(gray, 50, 150)
    
    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # === KEY FIX: Zero out border region before finding contours ===
    border_margin = 3
    fg_mask[:border_margin, :] = 0
    fg_mask[-border_margin:, :] = 0
    fg_mask[:, :border_margin] = 0
    fg_mask[:, -border_margin:] = 0
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cv2.Canny(gray, 50, 150)
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    
    # === ADDITIONAL FIX: Filter out contours that touch the border ===
    # Check if contour touches edges
    contour_points = largest.reshape(-1, 2)
    touches_left = np.any(contour_points[:, 0] <= border_margin)
    touches_right = np.any(contour_points[:, 0] >= w - border_margin - 1)
    touches_top = np.any(contour_points[:, 1] <= border_margin)
    touches_bottom = np.any(contour_points[:, 1] >= h - border_margin - 1)
    
    # Draw contour
    outline = np.zeros((h, w), np.uint8)
    cv2.drawContours(outline, [largest], -1, 255, 2)
    
    # === If contour touches border, zero out the border edges of the outline ===
    if touches_left or touches_right or touches_top or touches_bottom:
        # Remove any outline pixels along the borders (make it an open contour)
        border_clear = 5
        outline[:border_clear, :] = 0
        outline[-border_clear:, :] = 0
        outline[:, :border_clear] = 0
        outline[:, -border_clear:] = 0
    
    if dilate > 0:
        outline = cv2.dilate(outline, kernel, iterations=dilate)
    
    return outline


def get_patch_variants(patch, outline, allow_flip=True, allow_rotation=False, rotation_steps=4):
    """
    Generate variants of patch with flips and rotations.
    """
    flip_variants = [(patch, outline, '')]
    
    if allow_flip:
        flip_variants.append((cv2.flip(patch, 1), cv2.flip(outline, 1), 'flip_h'))
        flip_variants.append((cv2.flip(patch, 0), cv2.flip(outline, 0), 'flip_v'))
        flip_variants.append((cv2.flip(cv2.flip(patch, 0), 1), cv2.flip(cv2.flip(outline, 0), 1), 'flip_hv'))
    
    if allow_rotation:
        if rotation_steps <= 4:
            angles = [0, 90, 180, 270]
        else:
            angles = np.linspace(0, 360, rotation_steps, endpoint=False)
    else:
        angles = [0]
    
    variants = []
    
    for p_flip, o_flip, flip_name in flip_variants:
        for angle in angles:
            if angle == 0:
                p_rot = p_flip
                o_rot = o_flip
                rot_name = ''
            elif angle == 90:
                p_rot = cv2.rotate(p_flip, cv2.ROTATE_90_CLOCKWISE)
                o_rot = cv2.rotate(o_flip, cv2.ROTATE_90_CLOCKWISE)
                rot_name = 'rot90'
            elif angle == 180:
                p_rot = cv2.rotate(p_flip, cv2.ROTATE_180)
                o_rot = cv2.rotate(o_flip, cv2.ROTATE_180)
                rot_name = 'rot180'
            elif angle == 270:
                p_rot = cv2.rotate(p_flip, cv2.ROTATE_90_COUNTERCLOCKWISE)
                o_rot = cv2.rotate(o_flip, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rot_name = 'rot270'
            else:
                h, w = p_flip.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int(h * sin + w * cos)
                new_h = int(h * cos + w * sin)
                M[0, 2] += (new_w - w) / 2
                M[1, 2] += (new_h - h) / 2
                p_rot = cv2.warpAffine(p_flip, M, (new_w, new_h))
                o_rot = cv2.warpAffine(o_flip, M, (new_w, new_h))
                rot_name = f'rot{int(angle)}'
            
            if flip_name and rot_name:
                name = f'{rot_name}_{flip_name}'
            elif flip_name:
                name = flip_name
            elif rot_name:
                name = rot_name
            else:
                name = 'original'
            
            variants.append((p_rot, o_rot, name))
    
    return variants


def match_outlines(patch_outline, target_outline, min_scale=0.3, max_scale=2.0, scale_steps=20):
    """Match patch outline to target outline across scales."""
    ph, pw = patch_outline.shape[:2]
    th, tw = target_outline.shape[:2]
    
    scales = np.linspace(min_scale, max_scale, scale_steps)
    
    best_score = -1
    best_result = (0, 0, 1.0, -1)
    
    MIN_PIXELS = 30
    
    patch_pixels = np.sum(patch_outline > 0)
    
    for scale in scales:
        new_w = int(pw * scale)
        new_h = int(ph * scale)
        
        if new_w >= tw or new_h >= th or new_w < MIN_PIXELS or new_h < MIN_PIXELS:
            continue
        
        patch_scaled = cv2.resize(patch_outline, (new_w, new_h))
        patch_scaled_pixels = np.sum(patch_scaled > 0)
        
        if patch_scaled_pixels < 50:
            continue
        
        result = cv2.matchTemplate(target_outline, patch_scaled, cv2.TM_CCORR_NORMED)
        
        result_flat = result.flatten()
        top_indices = np.argsort(result_flat)[-10:]
        
        for idx in top_indices:
            y_idx = idx // result.shape[1]
            x_idx = idx % result.shape[1]
            score = result_flat[idx]
            
            target_region = target_outline[y_idx:y_idx+new_h, x_idx:x_idx+new_w]
            if target_region.shape[0] != new_h or target_region.shape[1] != new_w:
                continue
            
            target_pixels = np.sum(target_region > 0)
            
            if target_pixels < patch_scaled_pixels * 0.2:
                continue
            
            overlap = np.sum((patch_scaled > 0) & (target_region > 0))
            overlap_ratio = overlap / (patch_scaled_pixels + 1e-8)
            
            final_score = score * (1 + 0.0 * overlap_ratio)
            
            if final_score > best_score:
                best_score = final_score
                best_result = (x_idx, y_idx, scale, final_score)
    
    return best_result


def create_composite(patch, target, x, y, scale, blend_mode='soft', alpha=0.9, patch_mask=None):
    """Composite patch onto target at position.
    
    Args:
        patch_mask: Optional mask indicating valid pixels (non-black areas from rotation).
                   If None, will auto-detect black pixels as invalid.
    """
    result = target.copy()
    ph, pw = patch.shape[:2]
    th, tw = target.shape[:2]
    
    new_w = int(pw * scale)
    new_h = int(ph * scale)
    patch_scaled = cv2.resize(patch, (new_w, new_h))
    
    # Create or scale the validity mask (to exclude black rotation artifacts)
    if patch_mask is not None:
        validity_mask = cv2.resize(patch_mask, (new_w, new_h))
    else:
        # Auto-detect: pixels that are very dark (near black) are likely rotation artifacts
        # Use a threshold - if all channels are below threshold, consider it background
        gray = cv2.cvtColor(patch_scaled, cv2.COLOR_RGB2GRAY) if len(patch_scaled.shape) == 3 else patch_scaled
        validity_mask = (gray > 5).astype(np.float32)  # Threshold of 5 to catch near-black
        # Erode slightly to clean up edges
        kernel = np.ones((3, 3), np.uint8)
        validity_mask = cv2.erode(validity_mask, kernel, iterations=1)
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(tw, x + new_w), min(th, y + new_h)
    
    px1, py1 = max(0, -x), max(0, -y)
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return result
    
    patch_region = patch_scaled[py1:py2, px1:px2]
    target_region = result[y1:y2, x1:x2]
    mask_region = validity_mask[py1:py2, px1:px2]
    
    if blend_mode == 'replace':
        # Only replace valid pixels
        mask_3ch = np.stack([mask_region] * 3, axis=-1)
        result[y1:y2, x1:x2] = np.where(mask_3ch > 0.5, patch_region, target_region)
    elif blend_mode == 'alpha':
        # Blend with alpha, but only for valid pixels
        blended = cv2.addWeighted(target_region, 1 - alpha, patch_region, alpha, 0)
        mask_3ch = np.stack([mask_region] * 3, axis=-1)
        result[y1:y2, x1:x2] = (target_region * (1 - mask_3ch * alpha) + patch_region * mask_3ch * alpha).astype(np.uint8)
    elif blend_mode == 'soft':
        h, w = patch_region.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        feather = min(h, w) // 6
        if feather > 2:
            for i in range(feather):
                weight = i / feather
                mask[i, :] *= weight
                mask[h-1-i, :] *= weight
                mask[:, i] *= weight
                mask[:, w-1-i] *= weight
        # Combine feather mask with validity mask
        mask = mask * alpha * mask_region
        mask_3ch = np.stack([mask] * 3, axis=-1)
        blended = (target_region * (1 - mask_3ch) + patch_region * mask_3ch).astype(np.uint8)
        result[y1:y2, x1:x2] = blended
    
    return result


def resize_to_max(img, max_resolution):
    """Resize image so longest side is max_resolution."""
    if max_resolution is None:
        return img, 1.0
    h, w = img.shape[:2]
    if max(h, w) <= max_resolution:
        return img, 1.0
    if w > h:
        new_w = max_resolution
        new_h = int(h * max_resolution / w)
    else:
        new_h = max_resolution
        new_w = int(w * max_resolution / h)
    scale_factor = new_w / w
    return cv2.resize(img, (new_w, new_h)), scale_factor


def process_single_image(args):
    """Process a single image - used for parallel processing."""
    example, info, variants, min_scale, max_scale, scale_steps, max_resolution, patch_scale = args
    
    image = example["image"].numpy()
    label = example["label"].numpy()
    class_name = info.features["label"].int2str(label)
    
    image_proc, img_scale = resize_to_max(image, max_resolution)
    
    target_outline = get_subject_outline(image_proc)
    
    best_var_score = -1
    best_var_result = None
    
    for patch_var, outline_var, transform in variants:
        x, y, match_scale, score = match_outlines(
            outline_var, target_outline, 
            min_scale=min_scale, max_scale=max_scale, scale_steps=scale_steps
        )
        
        if score > best_var_score:
            best_var_score = score
            x_orig = int(x / img_scale)
            y_orig = int(y / img_scale)
            scale_orig = match_scale * (patch_scale / img_scale)
            
            best_var_result = {
                'image': image,
                'class_name': class_name,
                'x': x_orig,
                'y': y_orig,
                'scale': scale_orig,
                'score': score,
                'transform': transform
            }
    
    return best_var_result


def search_dataset(patch_path, num_samples=200, top_k=5, 
                   min_scale=0.3, max_scale=2.0, scale_steps=20,
                   allow_flip=True, allow_rotation=False, rotation_steps=4,
                   outer_only=True, max_resolution=480, num_workers=None):
    """
    Search Caltech101 for images that best match the patch outline.
    """
    patch = cv2.imread(patch_path)
    if patch is None:
        raise ValueError(f"Could not load patch: {patch_path}")
    
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    
    patch_proc, patch_scale = resize_to_max(patch_rgb, max_resolution)
    
    print("Extracting patch outline...")
    patch_outline = get_subject_outline(patch_proc)
    
    variants = get_patch_variants(patch_proc, patch_outline, allow_flip, allow_rotation, rotation_steps)
    print(f"Testing {len(variants)} patch variants (flips/rotations)")
    
    print("Loading Caltech101 dataset...")
    ds, info = tfds.load("caltech101", split="train", with_info=True, shuffle_files=True)
    ds_list = list(ds)
    
    if num_samples < len(ds_list):
        import random
        samples = random.sample(ds_list, num_samples)
    else:
        samples = ds_list
    
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)
    
    print(f"Searching {len(samples)} images with {num_workers} workers...")
    
    results = []
    
    if num_workers <= 1:
        for example in tqdm(samples):
            args = (example, info, variants, min_scale, max_scale, scale_steps, max_resolution, patch_scale)
            result = process_single_image(args)
            if result:
                results.append(result)
    else:
        args_list = [
            (example, info, variants, min_scale, max_scale, scale_steps, max_resolution, patch_scale)
            for example in samples
        ]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_image, args): i for i, args in enumerate(args_list)}
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
    
    results.sort(key=lambda r: r['score'], reverse=True)
    
    return results[:top_k], patch_rgb, patch_outline


def visualize_results(patch, patch_outline, results, blend_mode='soft', alpha=0.9, output_path="search_results.png"):
    """Visualize top matches."""
    n = len(results)
    fig, axes = plt.subplots(3, n + 1, figsize=(4 * (n + 1), 12))
    
    # Column 0: Patch
    axes[0, 0].imshow(patch)
    axes[0, 0].set_title("Patch")
    axes[0, 0].axis("off")
    
    axes[1, 0].imshow(patch_outline, cmap='gray')
    axes[1, 0].set_title("Patch Outline")
    axes[1, 0].axis("off")
    
    axes[2, 0].axis("off")
    
    # Columns 1-n: Results
    for i, result in enumerate(results):
        image = result['image']
        x, y, scale, score = result['x'], result['y'], result['scale'], result['score']
        class_name = result['class_name']
        transform = result['transform']
        
        # Apply transform to patch for composite
        patch_transformed = patch.copy()
        if 'flip_h' in transform:
            patch_transformed = cv2.flip(patch_transformed, 1)
        if 'flip_v' in transform:
            patch_transformed = cv2.flip(patch_transformed, 0)
        
        # Handle rotation
        if 'rot' in transform:
            rot_part = transform.split('_')[0] if '_' in transform else transform
            if rot_part == 'rot90':
                patch_transformed = cv2.rotate(patch_transformed, cv2.ROTATE_90_CLOCKWISE)
            elif rot_part == 'rot180':
                patch_transformed = cv2.rotate(patch_transformed, cv2.ROTATE_180)
            elif rot_part == 'rot270':
                patch_transformed = cv2.rotate(patch_transformed, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rot_part.startswith('rot'):
                angle = int(rot_part[3:])
                h, w = patch_transformed.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int(h * sin + w * cos)
                new_h = int(h * cos + w * sin)
                M[0, 2] += (new_w - w) / 2
                M[1, 2] += (new_h - h) / 2
                patch_transformed = cv2.warpAffine(patch_transformed, M, (new_w, new_h))
        
        # Get outline
        outline = get_subject_outline(image)
        
        # Row 1: Target image with patch composited on top (CHANGED FROM GREEN BOX)
        image_vis = create_composite(patch_transformed, image, x, y, scale, blend_mode, alpha)
        
        axes[0, i + 1].imshow(image_vis)
        axes[0, i + 1].set_title(f"#{i+1}: {class_name}\nScore: {score:.3f} \n scale: ({scale})({transform})")
        axes[0, i + 1].axis("off")
        
        # Row 2: Outline
        axes[1, i + 1].imshow(outline, cmap='gray')
        axes[1, i + 1].set_title("Outline")
        axes[1, i + 1].axis("off")
        
        # Row 3: Composite at 50% alpha for comparison
        composite = create_composite(patch_transformed, image, x, y, scale, 'alpha', 0.5)
        axes[2, i + 1].imshow(composite)
        axes[2, i + 1].set_title("Composite (50%)")
        axes[2, i + 1].axis("off")
    
    plt.suptitle("Best Matches from Caltech101", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    # ============================================================
    # CONFIGURATION - Edit these to match composite.py
    # ============================================================
    PATCH_FOLDER = "image_composer/image_patches"  # Folder containing patch images 
    PATCH_PATH = os.path.join(PATCH_FOLDER, "fish.jpg")      # Path to your patch image
    OUTPUT_PATH = "search_results.png"   # Output path
    
    MIN_SCALE = 0.1                 # Minimum scale to try
    MAX_SCALE = 2.0                 # Maximum scale to try
    SCALE_STEPS = 20                # Number of scale steps to try
    BLEND_MODE = "replace"             # Options: "soft", "alpha", "replace"
    ALPHA = 1                     # Blend alpha (0-1)
    
    # Transform options
    ALLOW_FLIP = True               # Try horizontal/vertical flips
    ALLOW_ROTATION = True          # Try rotations
    ROTATION_STEPS = 12              # Number of rotation angles (4=90° steps, 8=45° steps, 12=30° steps)
    
    # Matching options
    OUTER_ONLY = True               # Only match outer silhouette
    
    # Performance options
    MAX_RESOLUTION = 480            # Max resolution for processing
    
    # Search options
    NUM_SAMPLES = 1000               # Number of images to search
    TOP_K = 5                       # Number of top matches to show
    NUM_WORKERS = None              # Number of parallel workers (None = auto, 1 = single-threaded)
    
    # Debug options
    DEBUG_MODE = False              # Set True to only search 10 images for quick testing
    # ============================================================
    
    # Override for debug mode
    if DEBUG_MODE:
        NUM_SAMPLES = 10
        print("DEBUG MODE: Only searching 10 images")
    
    # Search
    results, patch, patch_outline = search_dataset(
        PATCH_PATH, 
        num_samples=NUM_SAMPLES, 
        top_k=TOP_K,
        min_scale=MIN_SCALE,
        max_scale=MAX_SCALE,
        scale_steps=SCALE_STEPS,
        allow_flip=ALLOW_FLIP,
        allow_rotation=ALLOW_ROTATION,
        rotation_steps=ROTATION_STEPS,
        outer_only=OUTER_ONLY,
        max_resolution=MAX_RESOLUTION,
        num_workers=NUM_WORKERS
    )
    
    # Print results
    print(f"\nTop {TOP_K} matches:")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['class_name']}: score={r['score']:.3f}, scale={r['scale']:.2f}, transform={r['transform']}")
    
    # Visualize
    visualize_results(patch, patch_outline, results, BLEND_MODE, ALPHA, OUTPUT_PATH)