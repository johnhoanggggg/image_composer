"""Disk cache for expensive per-image computations (segmentation masks)."""

import hashlib
import os

import numpy as np

CACHE_ROOT = os.environ.get(
    "IMAGE_COMPOSER_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "image_composer"),
)
MASK_CACHE_DIR = os.path.join(CACHE_ROOT, "masks")
DATASET_DIR = os.path.join(CACHE_ROOT, "datasets")


def array_hash(arr):
    """Content hash of an array.

    The previous implementation hashed only the first 4096 bytes, which is
    roughly the top two rows of a 320px image.  Any two photos sharing a flat
    background at the top -- exactly the product-style images this pipeline
    prefers -- collided and silently reused each other's mask.  blake2b over
    the whole buffer costs well under a millisecond at these sizes.
    """
    arr = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


_warned = set()


def _warn_once(message):
    """Caching is best-effort, but a silent failure just costs everyone time."""
    if message not in _warned:
        _warned.add(message)
        print(f"Warning: {message}")


def _mask_path(key):
    return os.path.join(MASK_CACHE_DIR, f"{key}.png")


def load_mask(img, tag):
    """Return a cached uint8 mask for `img` under `tag`, or None."""
    import cv2

    path = _mask_path(f"{array_hash(img)}_{tag}")
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask


def save_mask(img, tag, mask):
    """Persist a uint8 mask.

    Stored as PNG rather than npz: same lossless bytes, ~4x smaller on disk and
    an order of magnitude faster to read back than np.load on a compressed npz.
    """
    import cv2

    os.makedirs(MASK_CACHE_DIR, exist_ok=True)
    path = _mask_path(f"{array_hash(img)}_{tag}")
    # The .png must stay last: cv2.imwrite picks its codec from the extension,
    # so a name ending in .tmp raises and nothing is ever cached.
    tmp = f"{path}.{os.getpid()}.tmp.png"
    try:
        if not cv2.imwrite(tmp, mask):
            raise OSError(f"imwrite returned False for {tmp}")
        os.replace(tmp, path)
    except Exception as exc:
        _warn_once(f"mask cache disabled ({type(exc).__name__}: {exc})")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
