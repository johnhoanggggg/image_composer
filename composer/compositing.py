"""Placing a cut-out patch onto a canvas."""

import cv2
import numpy as np


def _blend_region(target_region, patch_region, mask_region, blend_mode, alpha):
    if blend_mode == "replace":
        m3 = mask_region[..., None]
        return np.where(m3 > 0.5, patch_region, target_region)

    if blend_mode == "alpha":
        m3 = (mask_region * alpha)[..., None]
        return (target_region * (1 - m3) + patch_region * m3).astype(np.uint8)

    # "soft": feather the rectangular border so the paste has no hard seam.
    h, w = patch_region.shape[:2]
    feather = min(h, w) // 6
    ramp = np.ones((h, w), np.float32)
    if feather > 2:
        edge = np.minimum(np.arange(h)[:, None], np.arange(w)[None, :])
        edge = np.minimum(edge, np.minimum((h - 1) - np.arange(h)[:, None],
                                           (w - 1) - np.arange(w)[None, :]))
        ramp = np.clip(edge / float(feather), 0.0, 1.0).astype(np.float32)
    m3 = (ramp * alpha * mask_region)[..., None]
    return (target_region * (1 - m3) + patch_region * m3).astype(np.uint8)


def composite(patch, target, x, y, scale, blend_mode="soft", alpha=0.9,
              patch_mask=None, return_placed_mask=False, clip_mask=None):
    """Paste `patch` onto a copy of `target` at (x, y), scaled by `scale`.

    `patch_mask` is a float32 alpha in [0, 1]; without one the patch's
    non-black pixels are used, which is only right for pre-cut images.

    `clip_mask` is a target-sized uint8 stencil the paste is confined to, used
    to keep objects inside the animal's silhouette.
    """
    result = target.copy()
    ph, pw = patch.shape[:2]
    th, tw = target.shape[:2]

    new_w, new_h = max(1, int(round(pw * scale))), max(1, int(round(ph * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    patch_scaled = cv2.resize(patch, (new_w, new_h), interpolation=interp)

    if patch_mask is not None:
        validity = cv2.resize(patch_mask.astype(np.float32), (new_w, new_h),
                              interpolation=interp)
    else:
        gray = cv2.cvtColor(patch_scaled, cv2.COLOR_RGB2GRAY)
        validity = (gray > 5).astype(np.float32)
        validity = cv2.erode(validity, np.ones((3, 3), np.uint8), iterations=1)

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(tw, x + new_w), min(th, y + new_h)
    px1, py1 = max(0, -x), max(0, -y)
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)

    placed = np.zeros((th, tw), np.uint8) if return_placed_mask else None
    if x2 <= x1 or y2 <= y1:
        return (result, placed) if return_placed_mask else result

    patch_region = patch_scaled[py1:py2, px1:px2]
    mask_region = np.clip(validity[py1:py2, px1:px2], 0.0, 1.0)

    if clip_mask is not None:
        stencil = clip_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        mask_region = mask_region * stencil

    result[y1:y2, x1:x2] = _blend_region(result[y1:y2, x1:x2], patch_region,
                                         mask_region, blend_mode, alpha)

    if return_placed_mask:
        placed[y1:y2, x1:x2] = (mask_region > 0.5).astype(np.uint8) * 255
        return result, placed
    return result
