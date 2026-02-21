#!/usr/bin/env python3
"""
Benchmark alternative outline-matching architectures against template matching.

Template matching (from search_caltech_rembg.py) is treated as ground truth.
Each alternative matcher implements the same interface:
    match_fn(patch_outline, target_outline, min_scale, max_scale, scale_steps) -> (x, y, scale, score)

Metrics reported:
  - Speed (ms per match)
  - Position agreement with ground truth (Euclidean distance in pixels)
  - Scale agreement (absolute difference)
  - Rank correlation (do they agree on which images are best matches?)
"""

import time
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tqdm import tqdm
from scipy import stats

# Import shared utilities from the main search module
from search_caltech_rembg import (
    resize_to_max,
    get_subject_outline_neural,
    get_patch_variants,
    precompute_outlines,
    match_outlines as match_outlines_pyramid,
    create_composite,
    _top_k_indices,
)


# ============================================================
# GROUND TRUTH: BRUTE-FORCE TEMPLATE MATCHING (all scales, full res)
# ============================================================

def match_outlines_bruteforce(patch_outline, target_outline,
                               min_scale=0.3, max_scale=2.0, scale_steps=20):
    """Exhaustive template matching at all scales at full resolution.
    This is the slowest but most accurate — used as ground truth baseline."""
    ph, pw = patch_outline.shape[:2]
    th, tw = target_outline.shape[:2]
    scales = np.linspace(min_scale, max_scale, scale_steps)

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
        top_indices = _top_k_indices(result_flat, 5)

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


# ============================================================
# MATCHER 1: CHAMFER DISTANCE
# ============================================================

def match_outlines_chamfer(patch_outline, target_outline, min_scale=0.3, max_scale=2.0, scale_steps=20):
    """Match using Chamfer distance on the distance transform.

    Instead of correlating raw pixel values, we:
    1. Compute the distance transform of the target outline (once)
    2. For each scale, overlay the patch outline on the DT and sum the distances
    3. Lower sum = better match (patch pixels land near target edges)

    This is faster because the distance transform is O(n) and the scoring is
    a simple masked sum rather than a full sliding-window correlation.
    """
    ph, pw = patch_outline.shape[:2]
    th, tw = target_outline.shape[:2]

    # Distance transform: each pixel = distance to nearest edge pixel
    # Invert because distanceTransform needs 0 = edge
    target_inv = cv2.bitwise_not(target_outline)
    dt = cv2.distanceTransform(target_inv, cv2.DIST_L2, 5)
    # Normalize so scores are comparable across image sizes
    dt_max = dt.max() if dt.max() > 0 else 1.0
    dt_norm = dt / dt_max

    scales = np.linspace(min_scale, max_scale, scale_steps)

    best_score = -1
    best_result = (0, 0, 1.0, -1)

    for scale in scales:
        new_w, new_h = int(pw * scale), int(ph * scale)

        if new_w >= tw or new_h >= th or new_w < 30 or new_h < 30:
            continue

        patch_scaled = cv2.resize(patch_outline, (new_w, new_h))
        patch_mask = (patch_scaled > 0)
        n_pixels = np.sum(patch_mask)

        if n_pixels < 50:
            continue

        # Sliding window using filter2D is much faster than a Python loop.
        # We convolve the DT with the patch mask — the result at each (x,y)
        # is the sum of DT values under the patch outline pixels.
        patch_kernel = patch_mask.astype(np.float32) / max(n_pixels, 1)
        response = cv2.filter2D(dt_norm, cv2.CV_32F, patch_kernel)

        # Crop to valid region (same as matchTemplate VALID output)
        valid_h = th - new_h + 1
        valid_w = tw - new_w + 1
        if valid_h <= 0 or valid_w <= 0:
            continue
        response_valid = response[new_h // 2:new_h // 2 + valid_h,
                                  new_w // 2:new_w // 2 + valid_w]

        if response_valid.size == 0:
            continue

        # Lower distance = better match, so we invert
        inv_response = 1.0 - response_valid
        result_flat = inv_response.flatten()
        top_indices = _top_k_indices(result_flat, 3)

        for idx in top_indices:
            y_idx = idx // inv_response.shape[1]
            x_idx = idx % inv_response.shape[1]
            score = result_flat[idx]

            # Validate there are target pixels in the region
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


# ============================================================
# MATCHER 2: HU MOMENTS PRE-FILTER + TEMPLATE
# ============================================================

def _hu_moments_from_outline(outline):
    """Extract log-Hu moments from a binary outline image."""
    contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(7)
    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    hu = cv2.HuMoments(moments).flatten()
    # Log-transform for scale invariance
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu_log


def match_outlines_hu_prefilter(patch_outline, target_outline,
                                 min_scale=0.3, max_scale=2.0, scale_steps=20):
    """Two-stage matcher:
    1. Quick Hu moments shape check — if shapes are very dissimilar, return low score
    2. If shapes pass the filter, run template matching but with fewer scale steps

    This saves time by skipping expensive template matching for clearly wrong shapes.
    """
    # Stage 1: Hu moments similarity check
    hu_patch = _hu_moments_from_outline(patch_outline)
    hu_target = _hu_moments_from_outline(target_outline)

    # Use L2 distance on log-Hu moments (scale/rotation invariant)
    hu_dist = np.linalg.norm(hu_patch - hu_target)

    # If shapes are extremely different, skip expensive matching
    # Return a low score proportional to how bad the shape match is
    if hu_dist > 15.0:
        return (0, 0, 1.0, max(0, 1.0 / (1.0 + hu_dist)))

    # Stage 2: Template matching with reduced scale steps (shapes are plausible)
    reduced_steps = max(5, scale_steps // 2)
    return match_outlines_bruteforce(patch_outline, target_outline,
                                     min_scale, max_scale, reduced_steps)


# Matcher 3 (pyramid 4x) is imported from search_caltech_rembg as match_outlines_pyramid


# ============================================================
# MATCHER 4: ORB FEATURE MATCHING
# ============================================================

def match_outlines_orb(patch_outline, target_outline,
                       min_scale=0.3, max_scale=2.0, scale_steps=20):
    """Keypoint-based matching using ORB features.

    Instead of sliding a window, we:
    1. Detect ORB keypoints on both outlines
    2. Match descriptors with BFMatcher
    3. Estimate scale and position from matched keypoints

    Extremely fast (no sliding window) but noisier for sparse outlines.
    """
    orb = cv2.ORB_create(nfeatures=200, scaleFactor=1.2, nlevels=8)

    kp1, des1 = orb.detectAndCompute(patch_outline, None)
    kp2, des2 = orb.detectAndCompute(target_outline, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return (0, 0, 1.0, -1)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 4:
        # Fall back: use all matches sorted by distance
        all_matches = bf.match(des1, des2)
        good = sorted(all_matches, key=lambda x: x.distance)[:min(20, len(all_matches))]

    if len(good) < 3:
        return (0, 0, 1.0, -1)

    # Estimate transform from matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Estimate scale from keypoint distances
    ph, pw = patch_outline.shape[:2]
    th, tw = target_outline.shape[:2]

    # Centroid of matched target points = estimated position
    cx = np.mean(dst_pts[:, 0, 0])
    cy = np.mean(dst_pts[:, 0, 1])

    # Estimate scale from spread of keypoints
    src_spread = np.std(src_pts[:, 0, :], axis=0).mean()
    dst_spread = np.std(dst_pts[:, 0, :], axis=0).mean()
    if src_spread > 1:
        est_scale = dst_spread / src_spread
    else:
        est_scale = 1.0

    est_scale = np.clip(est_scale, min_scale, max_scale)

    # Position: offset so patch center aligns with match centroid
    new_w, new_h = int(pw * est_scale), int(ph * est_scale)
    x = int(cx - new_w / 2)
    y = int(cy - new_h / 2)
    x = max(0, min(x, tw - new_w))
    y = max(0, min(y, th - new_h))

    # Score: ratio of good matches to total keypoints
    score = len(good) / max(len(kp1), 1)
    # Boost score if we have many matches
    score = min(score * 2.0, 1.0)

    return (x, y, est_scale, score)


# ============================================================
# BENCHMARK FRAMEWORK
# ============================================================

MATCHERS = {
    "template (ground truth)": match_outlines_bruteforce,
    "pyramid 4x":              match_outlines_pyramid,
    "chamfer distance":        match_outlines_chamfer,
    "hu prefilter + template": match_outlines_hu_prefilter,
    "orb features":            match_outlines_orb,
}


def benchmark_pair(patch_outline, target_outline, min_scale, max_scale, scale_steps):
    """Run all matchers on a single (patch, target) pair. Returns dict of results."""
    results = {}
    for name, fn in MATCHERS.items():
        t0 = time.perf_counter()
        x, y, scale, score = fn(patch_outline, target_outline,
                                min_scale, max_scale, scale_steps)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results[name] = {
            "x": x, "y": y, "scale": scale, "score": score,
            "time_ms": elapsed_ms,
        }
    return results


def run_benchmark(patch_path, num_samples=100, min_scale=0.3, max_scale=2.0,
                  scale_steps=20, max_resolution=320, seed=42):
    """Full benchmark: load data, run all matchers, compute metrics."""

    random.seed(seed)
    np.random.seed(seed)

    # Load and prepare patch
    patch = cv2.imread(patch_path)
    if patch is None:
        raise ValueError(f"Could not load patch: {patch_path}")
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch_proc, patch_scale = resize_to_max(patch_rgb, max_resolution)

    print("Extracting patch outline...")
    patch_outline = get_subject_outline_neural(patch_proc)

    # Load dataset
    print("Loading Caltech101 dataset...")
    ds, info = tfds.load("caltech101", split="train", with_info=True, shuffle_files=False)
    ds_list = list(ds)
    samples = random.sample(ds_list, min(num_samples, len(ds_list)))

    # Precompute outlines (uses disk cache)
    processed = precompute_outlines(samples, info, max_resolution, num_threads=1)

    # Filter out empty outlines
    processed = [p for p in processed if np.sum(p['outline'] > 0) >= 50]
    print(f"Benchmarking {len(processed)} images with valid outlines\n")

    # Run benchmark — store image data alongside matcher results
    all_results = []
    for data in tqdm(processed, desc="Benchmarking"):
        pair_results = benchmark_pair(
            patch_outline, data['outline'],
            min_scale, max_scale, scale_steps
        )
        pair_results["_class"] = data['class_name']
        pair_results["_image"] = data['image']
        pair_results["_outline"] = data['outline']
        all_results.append(pair_results)

    return all_results, patch_rgb, patch_outline, patch_proc


def compute_metrics(all_results):
    """Compute comparison metrics against template matching ground truth."""
    gt_name = "template (ground truth)"
    alt_names = [n for n in MATCHERS if n != gt_name]

    metrics = {}
    for alt in alt_names:
        pos_dists = []
        scale_diffs = []
        gt_scores = []
        alt_scores = []
        speedups = []

        for pair in all_results:
            gt = pair[gt_name]
            al = pair[alt]

            # Position distance
            dist = np.sqrt((gt["x"] - al["x"])**2 + (gt["y"] - al["y"])**2)
            pos_dists.append(dist)

            # Scale difference
            scale_diffs.append(abs(gt["scale"] - al["scale"]))

            # Scores for rank correlation
            gt_scores.append(gt["score"])
            alt_scores.append(al["score"])

            # Speedup
            if al["time_ms"] > 0:
                speedups.append(gt["time_ms"] / al["time_ms"])

        # Rank correlation (Spearman)
        if len(gt_scores) > 2:
            rank_corr, rank_p = stats.spearmanr(gt_scores, alt_scores)
        else:
            rank_corr, rank_p = 0.0, 1.0

        metrics[alt] = {
            "avg_pos_dist_px": np.mean(pos_dists),
            "median_pos_dist_px": np.median(pos_dists),
            "avg_scale_diff": np.mean(scale_diffs),
            "rank_correlation": rank_corr,
            "rank_p_value": rank_p,
            "avg_speedup": np.mean(speedups) if speedups else 0,
            "median_speedup": np.median(speedups) if speedups else 0,
            "avg_time_ms": np.mean([p[alt]["time_ms"] for p in all_results]),
            "gt_avg_time_ms": np.mean([p[gt_name]["time_ms"] for p in all_results]),
        }

    return metrics


def print_metrics(metrics):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Matcher':<28} {'Speedup':>8} {'Avg ms':>8} {'Pos Δ px':>10} "
          f"{'Scale Δ':>9} {'Rank ρ':>8}")
    print("=" * 90)

    for name, m in sorted(metrics.items(), key=lambda x: -x[1]["avg_speedup"]):
        print(f"{name:<28} {m['avg_speedup']:>7.1f}x {m['avg_time_ms']:>7.1f} "
              f"{m['avg_pos_dist_px']:>9.1f} {m['avg_scale_diff']:>9.3f} "
              f"{m['rank_correlation']:>8.3f}")

    print("=" * 90)
    gt_time = list(metrics.values())[0]["gt_avg_time_ms"]
    print(f"{'template (ground truth)':<28} {'1.0x':>8} {gt_time:>7.1f} "
          f"{'0.0':>10} {'0.000':>9} {'1.000':>8}")
    print()


def visualize_matches(all_results, patch, patch_outline, patch_proc,
                      num_images=6, output_path="matcher_visual_comparison.png"):
    """Show side-by-side composites: each row = one target image, each column = one matcher.

    This lets you visually compare where each matcher places the patch on the
    same target image, with template matching as ground truth in the first column.
    """
    matcher_names = list(MATCHERS.keys())
    n_matchers = len(matcher_names)

    # Pick the top images by ground truth score so we see interesting matches
    gt_name = "template (ground truth)"
    scored = [(i, r[gt_name]["score"]) for i, r in enumerate(all_results)
              if r[gt_name]["score"] > 0]
    scored.sort(key=lambda x: -x[1])
    picked = [all_results[i] for i, _ in scored[:num_images]]

    fig, axes = plt.subplots(num_images, n_matchers + 1, figsize=(4 * (n_matchers + 1), 4 * num_images))
    if num_images == 1:
        axes = axes[np.newaxis, :]

    # Column 0 header: original target
    axes[0, 0].set_title("Target image", fontsize=10, fontweight='bold')

    # Matcher column headers
    for col, name in enumerate(matcher_names):
        label = name if name != gt_name else "TEMPLATE (GT)"
        axes[0, col + 1].set_title(label, fontsize=9, fontweight='bold')

    for row, pair in enumerate(picked):
        target_image = pair["_image"]
        target_outline = pair["_outline"]
        class_name = pair["_class"]

        # Column 0: raw target image + outline overlay
        axes[row, 0].imshow(target_image)
        # Overlay outline in green
        outline_rgb = np.zeros_like(target_image)
        if target_outline.shape[:2] != target_image.shape[:2]:
            outline_disp = cv2.resize(target_outline, (target_image.shape[1], target_image.shape[0]))
        else:
            outline_disp = target_outline
        outline_rgb[outline_disp > 0] = [0, 255, 0]
        axes[row, 0].imshow(outline_rgb, alpha=0.35)
        axes[row, 0].set_ylabel(class_name, fontsize=9)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # Columns 1..N: composite from each matcher
        for col, name in enumerate(matcher_names):
            r = pair[name]
            x, y, scale, score = r["x"], r["y"], r["scale"], r["score"]

            if score > 0 and scale > 0:
                composite = create_composite(patch, target_image, x, y, scale,
                                             blend_mode='alpha', alpha=0.85)
            else:
                composite = target_image.copy()

            axes[row, col + 1].imshow(composite)
            time_ms = r["time_ms"]
            axes[row, col + 1].set_xlabel(
                f"score={score:.3f}  scale={scale:.2f}\n"
                f"pos=({x},{y})  {time_ms:.0f}ms",
                fontsize=7)
            axes[row, col + 1].set_xticks([])
            axes[row, col + 1].set_yticks([])

            # Highlight ground truth column with a border
            if name == gt_name:
                for spine in axes[row, col + 1].spines.values():
                    spine.set_edgecolor('#2196F3')
                    spine.set_linewidth(3)

    plt.suptitle("Visual Comparison: Patch Placement by Each Matcher\n"
                 "(Blue border = template matching ground truth)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visual comparison: {output_path}")
    plt.show()


def plot_comparison(all_results, metrics, output_path="benchmark_results.png"):
    """Create a visual comparison chart."""
    gt_name = "template (ground truth)"
    alt_names = sorted(metrics.keys(), key=lambda n: -metrics[n]["avg_speedup"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Speed comparison (bar chart)
    ax = axes[0, 0]
    names = [gt_name] + alt_names
    times = [metrics[alt_names[0]]["gt_avg_time_ms"]] + [metrics[n]["avg_time_ms"] for n in alt_names]
    colors = ["#2196F3"] + ["#4CAF50" if metrics[n]["avg_speedup"] > 1 else "#F44336" for n in alt_names]
    bars = ax.barh(names, times, color=colors)
    ax.set_xlabel("Average time per match (ms)")
    ax.set_title("Speed Comparison")
    ax.invert_yaxis()
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}ms", va='center', fontsize=9)

    # 2. Position accuracy (box plot)
    ax = axes[0, 1]
    pos_data = []
    labels = []
    for name in alt_names:
        dists = [np.sqrt((p[gt_name]["x"] - p[name]["x"])**2 +
                         (p[gt_name]["y"] - p[name]["y"])**2)
                 for p in all_results]
        pos_data.append(dists)
        labels.append(name)
    ax.boxplot(pos_data, labels=labels, vert=True)
    ax.set_ylabel("Position distance from GT (pixels)")
    ax.set_title("Position Accuracy vs Ground Truth")
    ax.tick_params(axis='x', rotation=20)

    # 3. Score correlation scatter
    ax = axes[1, 0]
    for name in alt_names:
        gt_scores = [p[gt_name]["score"] for p in all_results]
        alt_scores = [p[name]["score"] for p in all_results]
        ax.scatter(gt_scores, alt_scores, alpha=0.4, s=10, label=name)
    ax.set_xlabel("Template matching score (GT)")
    ax.set_ylabel("Alternative matcher score")
    ax.set_title("Score Correlation")
    ax.legend(fontsize=8)

    # 4. Speedup vs accuracy tradeoff
    ax = axes[1, 1]
    for name in alt_names:
        m = metrics[name]
        ax.scatter(m["avg_speedup"], m["avg_pos_dist_px"],
                   s=100, zorder=5)
        ax.annotate(name, (m["avg_speedup"], m["avg_pos_dist_px"]),
                    fontsize=8, ha='left', va='bottom')
    ax.scatter(1.0, 0.0, s=100, color="blue", marker="*", zorder=5)
    ax.annotate("template (GT)", (1.0, 0.0), fontsize=8, ha='left', va='bottom')
    ax.set_xlabel("Speedup vs template matching")
    ax.set_ylabel("Avg position error (px)")
    ax.set_title("Speed vs Accuracy Tradeoff")

    plt.suptitle("Matcher Benchmark: Alternatives vs Template Matching", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved benchmark chart: {output_path}")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    PATCH_FOLDER = "image_composer/image_patches"
    PATCH_PATH = os.path.join(PATCH_FOLDER, "fish.jpg")
    NUM_SAMPLES = 100         # Images to benchmark (keep reasonable for speed)
    MIN_SCALE = 0.1
    MAX_SCALE = 2.0
    SCALE_STEPS = 20
    MAX_RESOLUTION = 320

    print("Running matcher benchmark...")
    print(f"  Patch: {PATCH_PATH}")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Scales: {MIN_SCALE}-{MAX_SCALE} ({SCALE_STEPS} steps)")
    print()

    all_results, patch, patch_outline, patch_proc = run_benchmark(
        PATCH_PATH,
        num_samples=NUM_SAMPLES,
        min_scale=MIN_SCALE,
        max_scale=MAX_SCALE,
        scale_steps=SCALE_STEPS,
        max_resolution=MAX_RESOLUTION,
    )

    metrics = compute_metrics(all_results)
    print_metrics(metrics)
    plot_comparison(all_results, metrics)
    visualize_matches(all_results, patch, patch_outline, patch_proc, num_images=6)
