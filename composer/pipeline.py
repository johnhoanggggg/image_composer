"""Prepare images for matching: outline + alpha, computed once and reused."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

from . import segment
from .geometry import resize_to_max


def prepare_object(item, max_resolution, model_name=segment.DEFAULT_MODEL,
                   pad=2):
    """Segment one object image into everything the matcher needs.

    Work happens on a downscaled copy; the full-resolution image is kept for
    the final composite so output quality does not follow match resolution.

    Everything is cropped to the subject's bounding box.  Without that, a
    "scale" is a fraction of whatever empty frame the photographer left around
    the object, so the same requested scale yields wildly different apparent
    sizes.  Cropped, scale means the object's own size, and the min/max scale
    settings finally do what they say.
    """
    img = item["image"]
    proc, img_scale = resize_to_max(img, max_resolution)
    mask = segment.foreground_mask(proc, model_name)
    binary = segment.largest_component(segment.clean_mask(mask))
    outline = segment.mask_to_outline(mask)

    coverage = float(np.count_nonzero(binary)) / max(1, binary.size)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    bx, by, bw, bh = cv2.boundingRect(coords)
    ph, pw = proc.shape[:2]
    bx0, by0 = max(0, bx - pad), max(0, by - pad)
    bx1, by1 = min(pw, bx + bw + pad), min(ph, by + bh + pad)

    def crop(a):
        return a[by0:by1, bx0:bx1]

    return {
        "image": img,
        "bbox_proc": (bx0, by0, bx1 - bx0, by1 - by0),
        "proc": crop(proc),
        "img_scale": img_scale,
        "mask": crop(mask),
        "body": crop(binary),
        "outline": crop(outline),
        "coverage": coverage,
        "class_name": item.get("class_name", "object"),
        "source": item.get("source", ""),
    }


def prepare_objects(items, max_resolution, num_threads=4,
                    model_name=segment.DEFAULT_MODEL, progress=True,
                    min_coverage=0.02, max_coverage=0.92):
    """Segment a pool of objects, dropping ones with unusable silhouettes.

    A mask covering under `min_coverage` of the frame means rembg found nothing;
    over `max_coverage` means it kept the whole frame.  Neither can be cut out
    convincingly, and filtering them here saves the matcher from scoring them.
    """
    segment.get_session(model_name)  # load once, before threads start

    prepared = []
    dropped = 0

    def work(item):
        try:
            return prepare_object(item, max_resolution, model_name)
        except Exception as exc:
            print(f"  ! segmentation failed: {exc}")
            return None

    if num_threads <= 1:
        results = (work(i) for i in tqdm(items, disable=not progress))
        results = list(results)
    else:
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = [ex.submit(work, i) for i in items]
            for f in tqdm(as_completed(futures), total=len(futures),
                          disable=not progress):
                results.append(f.result())

    for r in results:
        if r is None:
            dropped += 1
            continue
        if not (min_coverage <= r["coverage"] <= max_coverage):
            dropped += 1
            continue
        if np.count_nonzero(r["outline"]) < 60:
            dropped += 1
            continue
        prepared.append(r)

    if dropped:
        print(f"  dropped {dropped} images with unusable silhouettes")
    return prepared
