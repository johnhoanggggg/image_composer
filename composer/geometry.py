"""Resizing, patch variants (flip/rotate) and transform replay."""

import cv2
import numpy as np


def resize_to_max(img, max_resolution):
    """Scale so the longest side is `max_resolution`. Returns (img, scale)."""
    if max_resolution is None:
        return img, 1.0
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_resolution:
        return img, 1.0
    scale = max_resolution / longest
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp), new_w / w


def _rotate(img, angle, flags=cv2.INTER_LINEAR):
    """Rotate about the centre, expanding the canvas to fit."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(img, M, (new_w, new_h), flags=flags)


def apply_transform(img, transform, interp=cv2.INTER_LINEAR):
    """Replay a transform name produced by `patch_variants` on any image."""
    if not transform or transform == "original":
        return img

    parts = transform.split("_")
    rot_part = parts[0] if parts[0].startswith("rot") else None
    flip_part = "_".join(parts[1:]) if rot_part else transform

    out = img
    if flip_part == "flip_h":
        out = cv2.flip(out, 1)
    elif flip_part == "flip_v":
        out = cv2.flip(out, 0)
    elif flip_part == "flip_hv":
        out = cv2.flip(cv2.flip(out, 0), 1)

    if rot_part:
        angle = float(rot_part[3:])
        if angle == 90:
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            out = cv2.rotate(out, cv2.ROTATE_180)
        elif angle == 270:
            out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle != 0:
            out = _rotate(out, angle, interp)
    return out


def crop_to_content(*images, reference=0, threshold=0):
    """Crop every image to the bounding box of `reference`'s non-zero pixels."""
    ref = images[reference]
    binary = (ref > threshold).astype(np.uint8)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return images
    x, y, w, h = cv2.boundingRect(coords)
    return tuple(im[y:y + h, x:x + w] for im in images)


def patch_variants(patch, outline, mask=None, allow_flip=True,
                   allow_rotation=False, rotation_steps=4):
    """Generate (patch, outline, mask, transform_name) variants.

    Non-cardinal rotations are cropped back to the outline's bounding box so a
    45-degree variant is not penalised for the black padding warpAffine adds.
    """
    if allow_rotation:
        if rotation_steps <= 4:
            angles = [0, 90, 180, 270][:max(1, rotation_steps)]
        else:
            angles = list(np.linspace(0, 360, rotation_steps, endpoint=False))
    else:
        angles = [0]

    # A vertical flip is a horizontal flip composed with a 180-degree rotation,
    # and flipping both axes *is* a 180-degree rotation.  So whenever the angle
    # set is closed under +180, half of the four flips reproduce variants the
    # rotations already cover -- 16 of 32 for the default settings -- and
    # generating them doubles the cost of every match for nothing.
    angle_set = {round(a % 360, 3) for a in angles}
    rotation_covers_180 = all(round((a + 180) % 360, 3) in angle_set
                              for a in angle_set)

    flips = [("", lambda im: im)]
    if allow_flip:
        flips.append(("flip_h", lambda im: cv2.flip(im, 1)))
        if not rotation_covers_180:
            flips += [
                ("flip_v", lambda im: cv2.flip(im, 0)),
                ("flip_hv", lambda im: cv2.flip(cv2.flip(im, 0), 1)),
            ]

    variants = []
    for flip_name, flip_fn in flips:
        p_flip, o_flip = flip_fn(patch), flip_fn(outline)
        m_flip = flip_fn(mask) if mask is not None else None

        for angle in angles:
            angle = float(angle)
            if angle == 0:
                p, o, m, rot_name = p_flip, o_flip, m_flip, ""
            elif angle in (90.0, 180.0, 270.0):
                code = {90.0: cv2.ROTATE_90_CLOCKWISE,
                        180.0: cv2.ROTATE_180,
                        270.0: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle]
                p = cv2.rotate(p_flip, code)
                o = cv2.rotate(o_flip, code)
                m = cv2.rotate(m_flip, code) if m_flip is not None else None
                rot_name = f"rot{int(angle)}"
            else:
                p = _rotate(p_flip, angle)
                o = _rotate(o_flip, angle, cv2.INTER_NEAREST)
                m = _rotate(m_flip, angle, cv2.INTER_NEAREST) if m_flip is not None else None
                if m is not None:
                    # Crop by the mask, not the outline: compose_animal has to
                    # reproduce this crop on the full-resolution image, where
                    # only the alpha mask is available.
                    m, p, o = crop_to_content(m, p, o, reference=0)
                else:
                    o, p = crop_to_content(o, p, reference=0)
                rot_name = f"rot{int(angle)}"

            if rot_name and flip_name:
                name = f"{rot_name}_{flip_name}"
            else:
                name = flip_name or rot_name or "original"
            variants.append((p, o, m, name))
    return variants
