#!/usr/bin/env python3
"""
Search Caltech101 dataset for best matching images to composite a patch onto.
Uses neural network-based segmentation (rembg) for outline extraction.
OPTIMIZED VERSION with GPU template matching, threading, disk caching,
coarse-to-fine scale search, and early termination.
"""

import os
import hashlib
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Neural network segmentation
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: rembg not installed. Install with: uv pip install rembg")

# Check for CUDA support in OpenCV
try:
    CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if CUDA_AVAILABLE:
        print(f"OpenCV CUDA available: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
except:
    CUDA_AVAILABLE = False
    print("OpenCV CUDA not available - using CPU for template matching")

# Global session for rembg (thread-safe with lock)
_rembg_session = None
_rembg_lock = threading.Lock()

# ============================================================
# DISK CACHE FOR SEGMENTATION MASKS
# ============================================================
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "image_composer_masks")


def _get_image_hash(img):
    """Fast hash of an image array for cache keying."""
    return hashlib.md5(img.tobytes()[:4096]).hexdigest()


def _get_cached_mask(img, max_resolution):
    """Load a cached mask from disk if it exists."""
    cache_key = f"{_get_image_hash(img)}_{max_resolution}"
    cache_path = os.path.join(_CACHE_DIR, f"{cache_key}.npz")
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            return data["mask"]
        except Exception:
            return None
    return None


def _save_cached_mask(img, max_resolution, mask):
    """Save a mask to disk cache."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_key = f"{_get_image_hash(img)}_{max_resolution}"
    cache_path = os.path.join(_CACHE_DIR, f"{cache_key}.npz")
    try:
        np.savez_compressed(cache_path, mask=mask)
    except Exception:
        pass


def get_rembg_session(model_name="isnet-general-use"):
    """Get or create rembg session (thread-safe)."""
    global _rembg_session
    if _rembg_session is None and REMBG_AVAILABLE:
        with _rembg_lock:
            if _rembg_session is None:  # Double-check after acquiring lock
                print(f"Loading segmentation model: {model_name}")
                _rembg_session = new_session(model_name)
    return _rembg_session


def get_mask_from_image(img, session):
    """Get segmentation mask for a single image."""
    pil_img = Image.fromarray(img)

    with _rembg_lock:  # rembg may not be thread-safe
        result = remove(pil_img, session=session, only_mask=True)

    if isinstance(result, Image.Image):
        mask = np.array(result)
    else:
        mask = result

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    return mask


def mask_to_outline(fg_mask, h, w, dilate=1, border_margin=3):
    """Convert a segmentation mask to an outline."""
    _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    fg_mask[:border_margin, :] = 0
    fg_mask[-border_margin:, :] = 0
    fg_mask[:, :border_margin] = 0
    fg_mask[:, -border_margin:] = 0
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros((h, w), np.uint8)
    
    largest = max(contours, key=cv2.contourArea)
    
    contour_points = largest.reshape(-1, 2)
    touches_border = (
        np.any(contour_points[:, 0] <= border_margin) or
        np.any(contour_points[:, 0] >= w - border_margin - 1) or
        np.any(contour_points[:, 1] <= border_margin) or
        np.any(contour_points[:, 1] >= h - border_margin - 1)
    )
    
    outline = np.zeros((h, w), np.uint8)
    cv2.drawContours(outline, [largest], -1, 255, 2)
    
    if touches_border:
        border_clear = 5
        outline[:border_clear, :] = 0
        outline[-border_clear:, :] = 0
        outline[:, :border_clear] = 0
        outline[:, -border_clear:] = 0
    
    if dilate > 0:
        outline = cv2.dilate(outline, kernel, iterations=dilate)
    
    return outline


def get_subject_outline_neural(img, dilate=1, border_margin=3):
    """Extract subject outline using neural network segmentation."""
    if not REMBG_AVAILABLE:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        return cv2.Canny(gray, 50, 150)
    
    h, w = img.shape[:2]
    session = get_rembg_session()
    fg_mask = get_mask_from_image(img, session)
    return mask_to_outline(fg_mask, h, w, dilate, border_margin)


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


def get_patch_variants(patch, outline, allow_flip=True, allow_rotation=False, rotation_steps=4):
    """Generate variants of patch with flips and rotations."""
    flip_variants = [(patch, outline, '')]
    
    if allow_flip:
        flip_variants.append((cv2.flip(patch, 1), cv2.flip(outline, 1), 'flip_h'))
        flip_variants.append((cv2.flip(patch, 0), cv2.flip(outline, 0), 'flip_v'))
        flip_variants.append((cv2.flip(cv2.flip(patch, 0), 1), cv2.flip(cv2.flip(outline, 0), 1), 'flip_hv'))
    
    if allow_rotation:
        angles = [0, 90, 180, 270] if rotation_steps <= 4 else np.linspace(0, 360, rotation_steps, endpoint=False)
    else:
        angles = [0]
    
    variants = []
    
    for p_flip, o_flip, flip_name in flip_variants:
        for angle in angles:
            if angle == 0:
                p_rot, o_rot, rot_name = p_flip, o_flip, ''
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
                cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
                new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
                M[0, 2] += (new_w - w) / 2
                M[1, 2] += (new_h - h) / 2
                p_rot = cv2.warpAffine(p_flip, M, (new_w, new_h))
                o_rot = cv2.warpAffine(o_flip, M, (new_w, new_h))
                # Crop to tight bounding box of the outline to remove
                # black padding that inflates the template size and
                # dilutes match scores vs cardinal rotations
                coords = cv2.findNonZero(o_rot)
                if coords is not None:
                    rx, ry, rw, rh = cv2.boundingRect(coords)
                    o_rot = o_rot[ry:ry+rh, rx:rx+rw]
                    p_rot = p_rot[ry:ry+rh, rx:rx+rw]
                rot_name = f'rot{int(angle)}'
            
            name = f'{rot_name}_{flip_name}' if (flip_name and rot_name) else (flip_name or rot_name or 'original')
            variants.append((p_rot, o_rot, name))
    
    return variants


# ============================================================
# PYRAMID 4x TEMPLATE MATCHING
# ============================================================

# Thread-local storage for GPU mats (each thread gets its own)
_thread_local = threading.local()


def get_gpu_matcher():
    """Get thread-local GPU template matcher."""
    if not hasattr(_thread_local, 'matcher'):
        _thread_local.matcher = cv2.cuda.createTemplateMatching(cv2.CV_8U, cv2.TM_CCORR_NORMED)
    return _thread_local.matcher


def _top_k_indices(arr, k):
    """O(n) top-k selection using argpartition instead of O(n log n) argsort."""
    if len(arr) <= k:
        return np.argsort(arr)[::-1]
    indices = np.argpartition(arr, -k)[-k:]
    # Sort just the top k
    return indices[np.argsort(arr[indices])[::-1]]


def _refine_at_full_res(patch_outline, target_outline, scales, ph, pw, th, tw, top_k=3):
    """Run full-resolution template matching at specific scales and return the best result."""
    best_score = -1
    best_result = (0, 0, 1.0, -1)

    for scale in scales:
        new_w, new_h = int(pw * scale), int(ph * scale)

        if new_w >= tw or new_h >= th or new_w < 30 or new_h < 30:
            continue

        patch_scaled = cv2.resize(patch_outline, (new_w, new_h))
        n_pixels = np.sum(patch_scaled > 0)
        if n_pixels < 50:
            continue

        result = cv2.matchTemplate(target_outline, patch_scaled, cv2.TM_CCORR_NORMED)
        result_flat = result.flatten()
        top_indices = _top_k_indices(result_flat, top_k)

        for idx in top_indices:
            y_idx = idx // result.shape[1]
            x_idx = idx % result.shape[1]
            score = result_flat[idx]

            target_region = target_outline[y_idx:y_idx + new_h, x_idx:x_idx + new_w]
            if target_region.shape != (new_h, new_w):
                continue
            target_pixels = np.sum(target_region > 0)
            if target_pixels < n_pixels * 0.2:
                continue

            if score > best_score:
                best_score = score
                best_result = (x_idx, y_idx, scale, score)

    return best_result


def match_outlines(patch_outline, target_outline, min_scale=0.1, max_scale=0.9, scale_steps=20):
    """Pyramid 4x matching: coarse pass at 1/4 resolution, refine top scales at full res.

    Scales are expressed as fractions of the target's longer side, not as
    multipliers on the patch.  e.g. min_scale=0.1 means the patch's longer
    side will be at least 10% of the target's longer side.

    1. Downsample both outlines by 4x
    2. Run template matching at all scales on the small images (very fast)
    3. Pick the top 3 scales
    4. Refine only those 3 scales at full resolution
    """
    ph, pw = patch_outline.shape[:2]
    th, tw = target_outline.shape[:2]

    # Convert target-relative scales to patch multipliers
    target_max = max(th, tw)
    patch_max = max(ph, pw)
    if patch_max == 0:
        return (0, 0, 1.0, -1)
    internal_min = min_scale * target_max / patch_max
    internal_max = max_scale * target_max / patch_max

    # --- Level 1: Downsample by 4x ---
    ds = 4
    target_small = cv2.resize(target_outline, (tw // ds, th // ds),
                              interpolation=cv2.INTER_AREA)
    patch_small = cv2.resize(patch_outline, (pw // ds, ph // ds),
                             interpolation=cv2.INTER_AREA)
    # Re-threshold after downsampling to keep binary
    _, target_small = cv2.threshold(target_small, 64, 255, cv2.THRESH_BINARY)
    _, patch_small = cv2.threshold(patch_small, 64, 255, cv2.THRESH_BINARY)

    ph_s, pw_s = patch_small.shape[:2]
    th_s, tw_s = target_small.shape[:2]

    scales = np.linspace(internal_min, internal_max, scale_steps)

    # Coarse pass at low resolution — sweep all scales cheaply
    coarse_results = []
    for scale in scales:
        new_w_s = int(pw_s * scale)
        new_h_s = int(ph_s * scale)

        if new_w_s >= tw_s or new_h_s >= th_s or new_w_s < 8 or new_h_s < 8:
            continue

        ps = cv2.resize(patch_small, (new_w_s, new_h_s))
        if np.sum(ps > 0) < 10:
            continue

        result = cv2.matchTemplate(target_small, ps, cv2.TM_CCORR_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        coarse_results.append((scale, max_val))

    if not coarse_results:
        return (0, 0, 1.0, -1)

    # Pick top 3 scales from coarse pass
    coarse_results.sort(key=lambda x: x[1], reverse=True)
    top_scales = [r[0] for r in coarse_results[:3]]

    # --- Level 2: Refine at full resolution with only the best scales ---
    return _refine_at_full_res(patch_outline, target_outline, top_scales,
                               ph, pw, th, tw, top_k=3)


# ============================================================
# COMPOSITING
# ============================================================

def create_composite(patch, target, x, y, scale, blend_mode='soft', alpha=0.9, patch_mask=None):
    """Composite patch onto target at position."""
    result = target.copy()
    ph, pw = patch.shape[:2]
    th, tw = target.shape[:2]
    
    new_w, new_h = int(pw * scale), int(ph * scale)
    patch_scaled = cv2.resize(patch, (new_w, new_h))
    
    if patch_mask is not None:
        validity_mask = cv2.resize(patch_mask, (new_w, new_h))
    else:
        gray = cv2.cvtColor(patch_scaled, cv2.COLOR_RGB2GRAY) if len(patch_scaled.shape) == 3 else patch_scaled
        validity_mask = (gray > 5).astype(np.float32)
        validity_mask = cv2.erode(validity_mask, np.ones((3, 3), np.uint8), iterations=1)
    
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
        mask_3ch = np.stack([mask_region] * 3, axis=-1)
        result[y1:y2, x1:x2] = np.where(mask_3ch > 0.5, patch_region, target_region)
    elif blend_mode == 'alpha':
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
        mask = mask * alpha * mask_region
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result[y1:y2, x1:x2] = (target_region * (1 - mask_3ch) + patch_region * mask_3ch).astype(np.uint8)
    
    return result


# ============================================================
# THREADED PROCESSING
# ============================================================

def precompute_single_outline(args):
    """Process a single image for outline extraction.
    Uses disk cache to skip rembg inference for previously seen images."""
    example, info, max_resolution = args

    image = example["image"].numpy()
    label = example["label"].numpy()
    class_name = info.features["label"].int2str(label)

    image_proc, img_scale = resize_to_max(image, max_resolution)

    # Try disk cache first
    cached_mask = _get_cached_mask(image_proc, max_resolution)
    if cached_mask is not None:
        h, w = image_proc.shape[:2]
        outline = mask_to_outline(cached_mask, h, w)
        return {
            'image': image,
            'outline': outline,
            'img_scale': img_scale,
            'class_name': class_name
        }

    # Cache miss — run rembg
    session = get_rembg_session()
    mask = get_mask_from_image(image_proc, session)
    h, w = image_proc.shape[:2]
    outline = mask_to_outline(mask, h, w)

    # Save to disk cache for next run
    _save_cached_mask(image_proc, max_resolution, mask)

    return {
        'image': image,
        'outline': outline,
        'img_scale': img_scale,
        'class_name': class_name
    }


def precompute_outlines(samples, info, max_resolution, num_threads=1):
    """Precompute all outlines with disk caching.
    Cached masks skip the neural net entirely, so subsequent runs are much faster."""
    print(f"Precomputing segmentation masks for {len(samples)} images...")

    # Pre-load the model once before any threads start
    get_rembg_session()

    processed = []

    if num_threads <= 1:
        for example in tqdm(samples):
            result = precompute_single_outline((example, info, max_resolution))
            processed.append(result)
    else:
        args_list = [(ex, info, max_resolution) for ex in samples]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(precompute_single_outline, args) for args in args_list]
            for future in tqdm(as_completed(futures), total=len(futures)):
                processed.append(future.result())

    return processed


def match_single_image(args):
    """Match a single image against all patch variants.
    Includes early termination: if a near-perfect match is found, skip remaining variants."""
    data, variants, min_scale, max_scale, scale_steps, patch_scale, early_stop_threshold = args

    target_outline = data['outline']
    img_scale = data['img_scale']

    # Skip images with empty outlines
    if np.sum(target_outline > 0) < 50:
        return None

    best_var_score = -1
    best_var_result = None

    for patch_var, outline_var, transform in variants:
        x, y, match_scale, score = match_outlines(
            outline_var, target_outline,
            min_scale=min_scale, max_scale=max_scale, scale_steps=scale_steps
        )

        if score > best_var_score:
            best_var_score = score
            best_var_result = {
                'image': data['image'],
                'class_name': data['class_name'],
                'x': int(x / img_scale),
                'y': int(y / img_scale),
                'scale': match_scale * (patch_scale / img_scale),
                'score': score,
                'transform': transform
            }

            # Early termination: very high score means we won't do much better
            if best_var_score >= early_stop_threshold:
                break

    return best_var_result


def match_all_images(processed_images, variants, min_scale, max_scale, scale_steps,
                     patch_scale, num_threads=4, early_stop_threshold=0.95):
    """Match outlines with threading. Includes early-stop threshold per image."""
    print(f"Matching {len(processed_images)} images against {len(variants)} variants using {num_threads} threads...")

    results = []
    args_list = [
        (data, variants, min_scale, max_scale, scale_steps, patch_scale, early_stop_threshold)
        for data in processed_images
    ]

    if num_threads <= 1:
        for args in tqdm(args_list):
            result = match_single_image(args)
            if result:
                results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(match_single_image, args) for args in args_list]
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)

    return results


# ============================================================
# MAIN SEARCH FUNCTION
# ============================================================

def _get_patch_fg_mask(img):
    """Get the foreground segmentation mask for a patch image using rembg.
    Returns a float32 mask in [0, 1] where 1 = foreground."""
    if not REMBG_AVAILABLE:
        # Fallback: treat non-black pixels as foreground
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        return (gray > 10).astype(np.float32)

    session = get_rembg_session()
    mask = get_mask_from_image(img, session)
    # Smooth edges for clean compositing
    mask_f = mask.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (5, 5), 0)
    return mask_f


def search_dataset(patch_path, num_samples=200, top_k=5,
                   min_scale=0.1, max_scale=0.9, scale_steps=20,
                   allow_flip=True, allow_rotation=False, rotation_steps=4,
                   max_resolution=320, num_threads_segmentation=1, num_threads_matching=4,
                   early_stop_threshold=0.95):
    """Search Caltech101 for images that best match the patch outline.

    Args:
        num_threads_segmentation: Threads for neural net (keep low, 1-2)
        num_threads_matching: Threads for template matching (can be higher, 4-8)
        early_stop_threshold: Skip remaining variants if a match score exceeds this (0-1)
    """

    # Load patch
    patch = cv2.imread(patch_path)
    if patch is None:
        raise ValueError(f"Could not load patch: {patch_path}")
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    patch_proc, patch_scale = resize_to_max(patch_rgb, max_resolution)

    print("Extracting patch outline...")
    patch_outline = get_subject_outline_neural(patch_proc)

    # Extract foreground mask for background-free compositing
    print("Extracting patch foreground mask...")
    patch_fg_mask = _get_patch_fg_mask(patch_rgb)

    variants = get_patch_variants(patch_proc, patch_outline, allow_flip, allow_rotation, rotation_steps)
    print(f"Testing {len(variants)} patch variants")

    # Load dataset
    print("Loading Caltech101 dataset...")
    ds, info = tfds.load("caltech101", split="train", with_info=True, shuffle_files=True)
    ds_list = list(ds)

    if num_samples < len(ds_list):
        import random
        samples = random.sample(ds_list, num_samples)
    else:
        samples = ds_list

    # Phase 1: Precompute segmentation masks (with disk caching)
    processed = precompute_outlines(samples, info, max_resolution, num_threads_segmentation)

    # Phase 2: Match outlines (pyramid 4x matching, benefits from threading)
    results = match_all_images(
        processed, variants, min_scale, max_scale, scale_steps, patch_scale,
        num_threads_matching, early_stop_threshold
    )

    # Sort and return top results
    results.sort(key=lambda r: r['score'], reverse=True)

    return results[:top_k], patch_rgb, patch_outline, patch_fg_mask


# ============================================================
# VISUALIZATION
# ============================================================

def _apply_transform(img, transform):
    """Apply flip/rotation transform string to an image or mask."""
    result = img.copy()
    if 'flip_h' in transform:
        result = cv2.flip(result, 1)
    if 'flip_v' in transform:
        result = cv2.flip(result, 0)

    if 'rot' in transform:
        rot_part = transform.split('_')[0] if '_' in transform else transform
        if rot_part == 'rot90':
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        elif rot_part == 'rot180':
            result = cv2.rotate(result, cv2.ROTATE_180)
        elif rot_part == 'rot270':
            result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rot_part.startswith('rot'):
            angle = int(rot_part[3:])
            h, w = result.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
            new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            result = cv2.warpAffine(result, M, (new_w, new_h))

    return result


def visualize_results(patch, patch_outline, results, patch_fg_mask=None,
                      blend_mode='soft', alpha=0.9, output_path="search_results.png"):
    """Visualize top matches as overlaid patches on target images in a square grid."""
    import math
    n = len(results)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))

    # Normalize axes to 2D array for uniform indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, result in enumerate(results):
        r, c = divmod(i, cols)
        image = result['image']
        x, y, scale, score = result['x'], result['y'], result['scale'], result['score']
        class_name = result['class_name']
        transform = result['transform']

        patch_transformed = _apply_transform(patch, transform)
        mask_transformed = _apply_transform(patch_fg_mask, transform) if patch_fg_mask is not None else None

        # For arbitrary rotations, get_patch_variants crops the rotated
        # outline to its tight bbox (removing padding).  The match
        # scale/position are relative to those cropped dimensions.
        # Replicate that crop here using the foreground mask so the
        # composite lines up.  Cardinal rotations (0/90/180/270) don't
        # crop during matching so we must NOT crop them here either.
        if mask_transformed is not None and 'rot' in transform:
            rot_part = transform.split('_')[0] if '_' in transform else transform
            if rot_part.startswith('rot') and rot_part not in ('rot90', 'rot180', 'rot270'):
                mask_binary = (mask_transformed > 0.1 if mask_transformed.dtype == np.float32
                               else mask_transformed > 25).astype(np.uint8) * 255
                coords = cv2.findNonZero(mask_binary)
                if coords is not None:
                    rx, ry, rw, rh = cv2.boundingRect(coords)
                    patch_transformed = patch_transformed[ry:ry+rh, rx:rx+rw]
                    mask_transformed = mask_transformed[ry:ry+rh, rx:rx+rw]

        image_vis = create_composite(patch_transformed, image, x, y, scale,
                                     blend_mode, alpha, patch_mask=mask_transformed)
        axes[r, c].imshow(image_vis)
        axes[r, c].set_title(f"#{i+1}: {class_name}\n{score:.3f} ({transform})", fontsize=8)
        axes[r, c].axis("off")

    # Hide empty cells
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    # ============================================================
    # CONFIGURATION
    # ============================================================
    PATCH_FOLDER = "image_composer/image_patches"
    PATCH_PATH = os.path.join(PATCH_FOLDER, "fish.jpg")
    OUTPUT_PATH = "search_results.png"
    
    MIN_SCALE = 0.1
    MAX_SCALE = 0.9
    SCALE_STEPS = 20
    BLEND_MODE = "replace"
    ALPHA = 1
    
    ALLOW_FLIP = True
    ALLOW_ROTATION = True
    ROTATION_STEPS = 16
    
    MAX_RESOLUTION = 480
    
    NUM_SAMPLES = 1000
    TOP_K = 5
    
    # Threading configuration
    NUM_THREADS_SEGMENTATION = 1  # Keep low (1-2), neural net is GPU-bound
    NUM_THREADS_MATCHING = 8      # Can be higher for template matching
    
    DEBUG_MODE = False
    # ============================================================
    
    if DEBUG_MODE:
        NUM_SAMPLES = 20
        print("DEBUG MODE: Only searching 20 images")
    
    if not REMBG_AVAILABLE:
        print("=" * 60)
        print("WARNING: rembg not installed!")
        print("Install with: uv pip install rembg")
        print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  CUDA available: {CUDA_AVAILABLE}")
    print(f"  Segmentation threads: {NUM_THREADS_SEGMENTATION}")
    print(f"  Matching threads: {NUM_THREADS_MATCHING}")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Variants: ~{4 * ROTATION_STEPS if ALLOW_ROTATION else 4}")
    print()
    
    results, patch, patch_outline, patch_fg_mask = search_dataset(
        PATCH_PATH,
        num_samples=NUM_SAMPLES,
        top_k=TOP_K,
        min_scale=MIN_SCALE,
        max_scale=MAX_SCALE,
        scale_steps=SCALE_STEPS,
        allow_flip=ALLOW_FLIP,
        allow_rotation=ALLOW_ROTATION,
        rotation_steps=ROTATION_STEPS,
        max_resolution=MAX_RESOLUTION,
        num_threads_segmentation=NUM_THREADS_SEGMENTATION,
        num_threads_matching=NUM_THREADS_MATCHING
    )

    print(f"\nTop {TOP_K} matches:")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['class_name']}: score={r['score']:.3f}, scale={r['scale']:.2f}, transform={r['transform']}")

    visualize_results(patch, patch_outline, results, patch_fg_mask, BLEND_MODE, ALPHA, OUTPUT_PATH)