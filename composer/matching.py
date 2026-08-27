"""Shape matching between a patch outline and a target.

Scoring
-------
The original pipeline scored placements with TM_CCORR_NORMED over two binary
contour images.  Binary contours are thin, so correlation is near-zero unless
edges land on each other pixel-exactly, and the normalisation term rewards
placements over dense clutter regardless of whether the shapes agree.

This module scores with a symmetric *chamfer* match instead.  A distance
transform of the target contours is turned into a proximity field
exp(-d / tau); correlating the patch's binary contour against that field and
dividing by the patch's edge count yields, exactly, the mean proximity of
patch edge pixels to the nearest target edge -- a real number in [0, 1] that
degrades gracefully with misalignment instead of falling off a cliff.  The
reverse direction (target edges near patch edges) is added so a small patch
cannot win by hiding inside a dense blob.

A containment term measures how much of the patch's *filled* silhouette lands
inside the target's subject mask, which is what keeps overlaid objects on the
animal rather than drifting onto the background.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

# Weights for the three score terms; they sum to 1 so scores stay in [0, 1].
W_PRECISION = 0.30   # patch edges land on target edges
W_RECALL = 0.50      # target contour explained by the patch
W_CONTAIN = 0.20     # patch body inside target subject

MIN_EDGE_PIXELS = 40
MIN_TEMPLATE_SIDE = 12


def proximity_field(outline, tau=6.0):
    """exp(-distance_to_nearest_edge / tau) as float32 in [0, 1]."""
    inverted = np.where(outline > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)
    return np.exp(-dist / float(tau)).astype(np.float32)


class TargetContext:
    """Everything about a target that can be precomputed once.

    Reused across every patch, every variant and every scale, which is what
    makes the recursive levels cheap.
    """

    __slots__ = ("outline", "proximity", "edges_f", "edge_integral", "subject",
                 "subject_integral", "occupied", "occupied_integral",
                 "shape", "edge_count", "subject_area", "_small")

    def __init__(self, outline, subject_mask=None, occupied_mask=None, tau=6.0):
        self._small = {}
        self.outline = outline
        self.shape = outline.shape[:2]
        self.proximity = proximity_field(outline, tau)
        self.edge_count = float(np.count_nonzero(outline))

        self.edges_f = (outline > 0).astype(np.float32)
        self.edge_integral = cv2.integral(self.edges_f)

        if subject_mask is None:
            subject_mask = np.zeros(self.shape, np.uint8)
        self.subject = (subject_mask > 127).astype(np.float32)
        self.subject_integral = cv2.integral(self.subject)
        self.subject_area = float(self.subject.sum())

        if occupied_mask is None:
            occupied = np.zeros(self.shape, np.float32)
        else:
            occupied = (occupied_mask > 127).astype(np.float32)
        self.occupied = occupied
        self.occupied_integral = cv2.integral(occupied)

    def update_occupied(self, occupied_mask):
        """Swap in a new occupancy mask, keeping the expensive fields.

        The proximity field, subject mask and edge integrals describe the
        animal and never change between rounds; only the record of what has
        already been pasted does.  Recomputing just that turns each extra round
        into one integral image instead of a full context rebuild.
        """
        occupied = (occupied_mask > 127).astype(np.float32)
        if occupied.shape != self.shape:
            occupied = cv2.resize(occupied, (self.shape[1], self.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        self.occupied = occupied
        self.occupied_integral = cv2.integral(occupied)
        for factor, small in self._small.items():
            small.occupied = (cv2.resize(
                occupied, (small.shape[1], small.shape[0]),
                interpolation=cv2.INTER_AREA) > 0.5).astype(np.float32)
            small.occupied_integral = cv2.integral(small.occupied)


def _edge_distance(edge_binary):
    """Euclidean distance from every pixel to the nearest set edge pixel."""
    inv = np.where(edge_binary > 0, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def _score_at_scale(ctx, patch_outline, patch_body, scale, patch_dist=None,
                    overlap_penalty=0.0, min_body_frac=0.0, tau=6.0,
                    max_overlap=1.0, min_containment=0.0):
    """Dense score map for one scale. Returns (best_score, x, y) or None."""
    ph, pw = patch_outline.shape[:2]
    th, tw = ctx.shape
    new_w, new_h = int(round(pw * scale)), int(round(ph * scale))

    if (new_w < MIN_TEMPLATE_SIDE or new_h < MIN_TEMPLATE_SIDE
            or new_w >= tw or new_h >= th):
        return None

    edge_t = cv2.resize(patch_outline, (new_w, new_h), interpolation=cv2.INTER_AREA)
    edge_t = (edge_t > 40).astype(np.float32)
    n_edge = float(edge_t.sum())
    if n_edge < MIN_EDGE_PIXELS:
        return None

    body_t = None
    n_body = 0.0
    if patch_body is not None:
        body_t = cv2.resize(patch_body, (new_w, new_h), interpolation=cv2.INTER_AREA)
        body_t = (body_t > 40).astype(np.float32)
        n_body = float(body_t.sum())

        # Size gate: an object covering a sliver of the animal is not a
        # composition, it is a speck.  Reject the scale outright rather than
        # letting the score decide, so results stay legible.
        if min_body_frac > 0 and ctx.subject_area > 0:
            if n_body < min_body_frac * ctx.subject_area:
                return None

    # Precision: mean proximity of this patch's edges to the target's edges.
    precision = cv2.matchTemplate(ctx.proximity, edge_t, cv2.TM_CCORR) / n_edge

    # Recall: the fraction of the target's *entire* contour lying close to one
    # of this patch's edges.  Correlating the target edge map against the
    # patch's own proximity field gives this exactly, and because the whole
    # contour is in the denominator, the term grows with a patch that traces
    # more of the animal -- which cancels precision's bias toward vanishingly
    # small placements.
    #
    # The patch's proximity field is derived by rescaling one distance
    # transform computed by the caller.  Shrinking a shape by s shrinks every
    # distance within it by s, so resampling the transform and multiplying by
    # the scale reproduces it, turning a per-scale distanceTransform (the most
    # expensive operation in the sweep) into a resize.
    if patch_dist is None:
        dist_t = _edge_distance(edge_t)
    else:
        dist_t = cv2.resize(patch_dist, (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR) * scale
    prox_t = np.ascontiguousarray(np.exp(-dist_t / float(tau)), dtype=np.float32)
    recall = cv2.matchTemplate(ctx.edges_f, prox_t, cv2.TM_CCORR) / max(ctx.edge_count, 1.0)

    score = W_PRECISION * precision + W_RECALL * recall

    if body_t is not None and n_body > 0:
        inside = cv2.matchTemplate(ctx.subject, body_t, cv2.TM_CCORR) / n_body
        score = score + W_CONTAIN * inside

        if min_containment > 0:
            # Objects belong inside the subject's outline, not hanging off it
            # into the background.  As a scoring term alone, containment loses
            # to a strong edge match and the object ends up half in the sky, so
            # it is a gate: at least this fraction of the object's body must
            # land on the subject.  It bounds where a placement sits without
            # trimming the object, which is what keeps the edges ragged --
            # the leftover fraction is exactly the overhang that shows.
            score = np.where(inside < min_containment, -np.inf, score)

        if ctx.occupied.any():
            covered = cv2.matchTemplate(ctx.occupied, body_t, cv2.TM_CCORR) / n_body
            if overlap_penalty > 0:
                score = score - overlap_penalty * covered
            if max_overlap < 1.0:
                # A hard gate, not just a penalty.  A late object that scores
                # well can otherwise outbid the penalty and bury everything
                # placed before it, which is how a wristwatch ends up covering
                # the entire head.  Placements burying more than max_overlap of
                # their own body in already-placed pixels are simply illegal.
                score = np.where(covered > max_overlap, -np.inf, score)

    idx = int(np.argmax(score))
    y, x = divmod(idx, score.shape[1])
    best = float(score[y, x])
    if not np.isfinite(best):
        return None
    return best, x, y


def match_outline(ctx, patch_outline, patch_body, min_scale=0.3, max_scale=0.8,
                  scale_steps=20, max_resolution=320, coarse_factor=3,
                  refine_top=3, overlap_penalty=0.0, min_body_frac=0.0,
                  tau=6.0, max_overlap=1.0, rank_only=False,
                  min_containment=0.0):
    """Coarse-to-fine scale sweep. Returns dict or None.

    Scales are expressed as a fraction of `max_resolution`, so min_scale=0.3
    means "the patch's longer side covers at least 30% of a 320px canvas",
    independent of either image's pixel dimensions.
    """
    ph, pw = patch_outline.shape[:2]
    patch_max = max(ph, pw)
    if patch_max == 0:
        return None

    lo = min_scale * max_resolution / patch_max
    hi = max_scale * max_resolution / patch_max
    if hi <= 0:
        return None
    scales = np.linspace(lo, hi, max(1, scale_steps))

    # One distance transform for the whole sweep, rescaled per scale.
    patch_dist = _edge_distance(patch_outline > 0)

    # Coarse pass on downsampled copies: the score surface is smooth in scale,
    # so the ranking survives a 3x reduction at ~9x the speed.
    cf = max(1, int(coarse_factor))
    if cf > 1 and min(ctx.shape) // cf >= 32:
        small_ctx = _downscale_context(ctx, cf)
        small_outline = cv2.resize(patch_outline,
                                   (max(1, pw // cf), max(1, ph // cf)),
                                   interpolation=cv2.INTER_AREA)
        small_body = (cv2.resize(patch_body, (max(1, pw // cf), max(1, ph // cf)),
                                 interpolation=cv2.INTER_AREA)
                      if patch_body is not None else None)
        small_dist = _edge_distance(small_outline > 0)
        coarse = []
        for s in scales:
            r = _score_at_scale(small_ctx, small_outline, small_body, s,
                                small_dist, overlap_penalty, min_body_frac,
                                tau=tau / cf, max_overlap=max_overlap,
                                min_containment=min_containment)
            if r is not None:
                coarse.append((r[0], s, r[1], r[2]))
        if not coarse:
            return None
        coarse.sort(key=lambda c: c[0], reverse=True)

        if rank_only:
            # Ranking pass: the coarse score alone decides which objects are
            # worth a full-resolution look.  Refining every object in a large
            # pool costs ~15x more than ranking it, and the ranking is only
            # used to pick which handful to refine.
            best_score, best_scale, bx, by = coarse[0]
            return {"score": best_score, "x": bx * cf, "y": by * cf,
                    "match_scale": best_scale}

        candidate_scales = [s for _, s, _, _ in coarse[:refine_top]]
    else:
        if rank_only:
            scales = scales[::max(1, len(scales) // 6)]
        candidate_scales = list(scales)

    best = None
    for s in candidate_scales:
        r = _score_at_scale(ctx, patch_outline, patch_body, s, patch_dist,
                            overlap_penalty, min_body_frac, tau=tau,
                            max_overlap=max_overlap,
                            min_containment=min_containment)
        if r is None:
            continue
        if best is None or r[0] > best[0]:
            best = (r[0], r[1], r[2], s)

    if best is None:
        return None
    return {"score": best[0], "x": best[1], "y": best[2], "match_scale": best[3]}


def _downscale_context(ctx, factor):
    """Cheap reduced-resolution view of a TargetContext for the coarse pass.

    Memoised on the context: every patch and every variant matched against this
    target reuses the same reduced copy instead of rebuilding four integral
    images per call.
    """
    cached = ctx._small.get(factor)
    if cached is not None:
        return cached
    th, tw = ctx.shape
    small = object.__new__(TargetContext)
    small.shape = (th // factor, tw // factor)
    small.outline = cv2.resize(ctx.outline, (small.shape[1], small.shape[0]),
                               interpolation=cv2.INTER_AREA)
    small.proximity = cv2.resize(ctx.proximity, (small.shape[1], small.shape[0]),
                                 interpolation=cv2.INTER_AREA)
    small.edge_count = float(np.count_nonzero(small.outline))
    small.edges_f = (small.outline > 0).astype(np.float32)
    small.edge_integral = cv2.integral(small.edges_f)
    # Re-binarise after downsampling.  subject and occupied are 0/1 masks, and
    # INTER_AREA turns their edges into a fractional ramp -- 37% of the non-zero
    # pixels at factor 6.  Fractions there make the containment and overlap
    # ratios mean something different at each resolution, and a placement that
    # clears the gates at full res gets rejected during ranking.
    small.subject = (cv2.resize(ctx.subject, (small.shape[1], small.shape[0]),
                                interpolation=cv2.INTER_AREA)
                     > 0.5).astype(np.float32)
    small.subject_integral = cv2.integral(small.subject)
    small.subject_area = float(small.subject.sum())
    small.occupied = (cv2.resize(ctx.occupied, (small.shape[1], small.shape[0]),
                                 interpolation=cv2.INTER_AREA)
                      > 0.5).astype(np.float32)
    small.occupied_integral = cv2.integral(small.occupied)
    small._small = {}
    ctx._small[factor] = small
    return small


def match_variants(ctx, variants, config, overlap_penalty=0.0,
                   min_body_frac=0.0, max_overlap=1.0, rank_only=False,
                   min_containment=0.0):
    """Best placement of any variant of one patch on one target.

    With rank_only the search stays at coarse resolution and skips the
    full-resolution refinement, which is the cheap first stage of the
    two-stage cascade in compose().
    """
    coarse_factor = (config.get("rank_coarse_factor", 6) if rank_only
                     else config.get("coarse_factor", 3))
    scale_steps = (config.get("rank_scale_steps", 8) if rank_only
                   else config["scale_steps"])

    if rank_only:
        # Ranking only has to decide whether an object is worth a closer look,
        # and neighbouring rotations of the same object score almost alike.
        # Sampling every Nth orientation halves the ranking cost again; the
        # promoted objects are then matched over every orientation.
        stride = max(1, int(config.get("rank_rotation_stride", 2)))
        variants = variants[::stride]

        # Gate loosely here.  A mask reduced 6x cannot resolve these ratios
        # exactly, and the cascade only ever refines what ranking shortlists,
        # so a strict gate at this stage throws away placements that would have
        # passed at full resolution -- and the round then reports that nothing
        # fits at all.  Full resolution applies the real gates.
        slack = float(config.get("rank_gate_slack", 0.75))
        min_containment = min_containment * slack
        max_overlap = min(1.0, max_overlap + (1.0 - slack))
        min_body_frac = min_body_frac * slack

    best = None
    for patch_img, outline, body, transform in variants:
        r = match_outline(
            ctx, outline, body,
            min_scale=config["min_scale"], max_scale=config["max_scale"],
            scale_steps=scale_steps,
            max_resolution=config["max_resolution"],
            coarse_factor=coarse_factor,
            refine_top=config.get("refine_top", 3),
            rank_only=rank_only,
            overlap_penalty=overlap_penalty,
            min_body_frac=min_body_frac,
            max_overlap=max_overlap,
            min_containment=min_containment,
        )
        if r is None:
            continue
        r["transform"] = transform
        if best is None or r["score"] > best["score"]:
            best = r
            if best["score"] >= config.get("early_stop_threshold", 1.1):
                break
    return best


def match_targets(targets, variants, config, num_threads=4, progress=True):
    """Match one patch's variants against many targets.

    Each target carries a prebuilt TargetContext under 'ctx'.
    """
    results = []

    def work(target):
        r = match_variants(target["ctx"], variants, config)
        if r is None:
            return None
        r = dict(r)
        img_scale = target["img_scale"]
        r["image"] = target["image"]
        r["class_name"] = target["class_name"]
        r["target_outline"] = target["ctx"].outline
        r["img_scale"] = img_scale
        # Positions and scales are computed on the downscaled working copy;
        # promote them to full-resolution coordinates for compositing.
        r["x"] = int(round(r["x"] / img_scale))
        r["y"] = int(round(r["y"] / img_scale))
        r["scale"] = r["match_scale"] * (config["patch_scale"] / img_scale)
        return r

    if num_threads <= 1:
        iterator = tqdm(targets, disable=not progress)
        for t in iterator:
            r = work(t)
            if r:
                results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = [ex.submit(work, t) for t in targets]
            for f in tqdm(as_completed(futures), total=len(futures),
                          disable=not progress):
                r = f.result()
                if r:
                    results.append(r)
    return results
