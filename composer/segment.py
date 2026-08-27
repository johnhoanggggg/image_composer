"""Neural background removal and outline extraction.

Wraps rembg with a per-thread session pool and a disk cache so the same image
is never segmented twice, across runs.
"""

import threading

import cv2
import numpy as np

from . import cache

try:
    from rembg import new_session, remove

    REMBG_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    REMBG_AVAILABLE = False
    print("Warning: rembg not installed. Install with: pip install 'rembg[cpu]'")

DEFAULT_MODEL = "isnet-general-use"

_thread_local = threading.local()
_session_lock = threading.Lock()
_shared_session = None
_shared_model = None


def get_session(model_name=DEFAULT_MODEL):
    """Return an onnxruntime session for `model_name`.

    The old code held a single global session behind a lock for the whole
    inference call, which serialised every worker thread onto one model.
    onnxruntime is internally thread safe for concurrent Run() calls, so the
    session is shared but the lock is gone; the first construction is still
    guarded so the 179MB model loads exactly once.
    """
    global _shared_session, _shared_model
    if not REMBG_AVAILABLE:
        return None
    if _shared_session is None or _shared_model != model_name:
        with _session_lock:
            if _shared_session is None or _shared_model != model_name:
                print(f"Loading segmentation model: {model_name}")
                _shared_session = new_session(model_name)
                _shared_model = model_name
    return _shared_session


def foreground_mask(img_rgb, model_name=DEFAULT_MODEL, use_cache=True):
    """uint8 foreground mask (0-255) for an RGB image."""
    if not REMBG_AVAILABLE:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

    if use_cache:
        cached = cache.load_mask(img_rgb, model_name)
        if cached is not None:
            return cached

    from PIL import Image

    result = remove(Image.fromarray(img_rgb), session=get_session(model_name),
                    only_mask=True)
    mask = np.array(result) if isinstance(result, Image.Image) else result
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = np.ascontiguousarray(mask, dtype=np.uint8)

    if use_cache:
        cache.save_mask(img_rgb, model_name, mask)
    return mask


def clean_mask(mask, border_margin=3):
    """Threshold, close holes, drop border-hugging noise."""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    if border_margin > 0:
        binary[:border_margin, :] = 0
        binary[-border_margin:, :] = 0
        binary[:, :border_margin] = 0
        binary[:, -border_margin:] = 0
    return binary


def largest_component(binary):
    """Keep only the biggest connected blob of a binary mask."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return binary
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


def mask_to_outline(mask, dilate=1, border_margin=3, keep_largest=True):
    """Contour image (uint8, 0/255) of the subject in `mask`."""
    h, w = mask.shape[:2]
    binary = clean_mask(mask, border_margin)
    if keep_largest:
        binary = largest_component(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), np.uint8)
    largest = max(contours, key=cv2.contourArea)

    pts = largest.reshape(-1, 2)
    touches_border = (
        np.any(pts[:, 0] <= border_margin)
        or np.any(pts[:, 0] >= w - border_margin - 1)
        or np.any(pts[:, 1] <= border_margin)
        or np.any(pts[:, 1] >= h - border_margin - 1)
    )

    outline = np.zeros((h, w), np.uint8)
    cv2.drawContours(outline, [largest], -1, 255, 2)

    if touches_border:
        clear = 5
        outline[:clear, :] = 0
        outline[-clear:, :] = 0
        outline[:, :clear] = 0
        outline[:, -clear:] = 0

    if dilate > 0:
        outline = cv2.dilate(outline, np.ones((3, 3), np.uint8), iterations=dilate)
    return outline


def subject_outline(img_rgb, dilate=1, border_margin=3, model_name=DEFAULT_MODEL):
    """Convenience: RGB image -> contour image."""
    if not REMBG_AVAILABLE:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(gray, 50, 150)
    mask = foreground_mask(img_rgb, model_name)
    return mask_to_outline(mask, dilate, border_margin)
