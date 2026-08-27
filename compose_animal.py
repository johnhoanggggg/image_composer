#!/usr/bin/env python3
"""Build an animal out of photographs of other objects.

The animal photo is the canvas.  Each round, every object in the pool is
scored against the canvas's contours and the best-fitting one is composited on
top; the next round matches against what is left uncovered, so a run of three
or four rounds lays down three or four different objects across the animal and
the silhouette still reads as the animal.

    python compose_animal.py --animal image_patches/duck3.png --rounds 4

Run with no arguments to use the defaults at the bottom of this file.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from composer import segment, sources
from composer.compositing import composite
from composer.geometry import apply_transform, crop_to_content, patch_variants, resize_to_max
from composer.matching import TargetContext, match_variants
from composer.pipeline import prepare_objects


def build_variants(obj, config):
    """Flip/rotate variants of a prepared object, at match resolution."""
    return patch_variants(
        obj["proc"], obj["outline"], obj["body"],
        allow_flip=config["allow_flip"],
        allow_rotation=config["allow_rotation"],
        rotation_steps=config["rotation_steps"],
    )


def describe_silhouette(raw_mask, subject):
    """Report how cleanly the animal separated, and warn when it did not.

    The whole pipeline assumes one dominant subject the background remover can
    isolate.  A cluttered scene segments into a fragmented blob, and the run
    then produces slivers rather than a composition -- worth saying up front
    rather than leaving the user to infer it from the output.
    """
    coverage = np.count_nonzero(subject) / subject.size
    cleaned = segment.clean_mask(raw_mask)
    total = float(np.count_nonzero(cleaned))
    kept = float(np.count_nonzero(subject))
    dominance = kept / total if total else 0.0

    hull_solidity = 1.0
    contours, _ = cv2.findContours(subject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(biggest))
        if hull_area > 0:
            hull_solidity = cv2.contourArea(biggest) / hull_area

    print(f"Animal silhouette: {coverage:.1%} of frame, "
          f"{dominance:.0%} of foreground, solidity {hull_solidity:.2f}")

    if coverage < 0.05:
        print("  ! the subject is tiny -- crop closer to the animal")
    elif coverage > 0.9:
        print("  ! almost the whole frame segmented as subject; the background "
              "remover found no clear edge")
    if dominance < 0.6:
        print("  ! the photo holds several subjects; only the largest is used. "
              "A single-animal photo on a plain background works far better.")
    return coverage


def is_cardinal(transform):
    """True when the transform involves no arbitrary-angle rotation."""
    rot = transform.split("_")[0]
    if not rot.startswith("rot"):
        return True
    return rot[3:] in ("", "0", "90", "180", "270")


def place_object(canvas, obj, result, blend_mode, alpha, clip_mask=None):
    """Composite the winning object onto the canvas at full resolution.

    The match ran on a downscaled, bbox-cropped copy, so the same crop and the
    same transform are replayed here on the original pixels; only then is the
    result scaled up, which keeps the output as sharp as the source photo
    rather than as sharp as the match resolution.
    """
    transform = result["transform"]

    bx, by, bw, bh = obj["bbox_proc"]
    s = obj["img_scale"]
    fh, fw = obj["image"].shape[:2]
    x0, y0 = int(round(bx / s)), int(round(by / s))
    x1, y1 = min(fw, int(round((bx + bw) / s))), min(fh, int(round((by + bh) / s)))
    img = obj["image"][y0:y1, x0:x1]
    if img.size == 0:
        img = obj["image"]

    # The cleaned, largest-component mask -- not the raw rembg output.  Raw
    # masks carry mid-range values across background pixels, and alpha blending
    # turns those into a pale halo of the object's original background.
    alpha_mask = cv2.resize(obj["body"], (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    alpha_mask = cv2.GaussianBlur(alpha_mask, (5, 5), 0).astype(np.float32) / 255.0

    img = apply_transform(img, transform)
    alpha_mask = apply_transform(alpha_mask, transform, cv2.INTER_LINEAR)

    if not is_cardinal(transform):
        # patch_variants cropped the rotated variant to its mask bbox; do the
        # same here or the placement lands offset by the rotation padding.
        binary = (alpha_mask > 0.1).astype(np.uint8) * 255
        binary, img, alpha_mask = crop_to_content(binary, img, alpha_mask,
                                                  reference=0)

    return composite(img, canvas, result["x"], result["y"], result["scale"],
                     blend_mode, alpha, patch_mask=alpha_mask,
                     return_placed_mask=True, clip_mask=clip_mask)


def compose(animal_path, objects, config, rounds=4, blend_mode="alpha",
            alpha=1.0, output_dir="output", save_steps=True, threads=4,
            clip_to_animal=True, allow_repeat_class=False,
            variant_cache=None):
    """Layer `rounds` objects onto the animal, best fit first."""
    animal_bgr = cv2.imread(animal_path, cv2.IMREAD_COLOR)
    if animal_bgr is None:
        raise SystemExit(f"Could not read animal image: {animal_path}")
    canvas = cv2.cvtColor(animal_bgr, cv2.COLOR_BGR2RGB)
    animal_name = os.path.splitext(os.path.basename(animal_path))[0]

    # The animal's own silhouette, fixed once. Later rounds must keep matching
    # against the animal, not against the collage growing on top of it.
    proc, _ = resize_to_max(canvas, config["max_resolution"])
    animal_mask = segment.foreground_mask(proc)
    # Keep the single largest blob.  subject_outline already traces only the
    # largest contour, so leaving every component in the subject mask made the
    # two disagree: objects were scored against one animal and then clipped to
    # a mask that also held background clutter, leaving slivers behind.
    animal_subject = segment.largest_component(segment.clean_mask(animal_mask))
    animal_outline = segment.subject_outline(proc)

    describe_silhouette(animal_mask, animal_subject)

    # Full-resolution copy of the same silhouette.  Clipping every paste to it
    # is what stops an object from spilling across the background and burying
    # the animal -- the failure mode where a well-scored watch simply covers
    # the whole head.  The animal's outline survives no matter what lands on it.
    animal_subject_full = cv2.resize(animal_subject,
                                     (canvas.shape[1], canvas.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
    if clip_to_animal:
        animal_subject_full = cv2.GaussianBlur(animal_subject_full, (5, 5), 0)

    occupied_full = np.zeros(canvas.shape[:2], np.uint8)
    history = [canvas.copy()]
    placed = []

    # Variants depend only on the object pool, so a caller composing several
    # canvases against the same pool can build them once and pass them in.
    if variant_cache is None:
        print(f"\nPrecomputing variants for {len(objects)} objects...")
        variant_cache = [build_variants(o, config) for o in objects]
        n_variants = sum(len(v) for v in variant_cache)
        print(f"  {n_variants} variants total "
              f"({n_variants / max(1, len(objects)):.0f} per object)")

    used = set()
    used_classes = set()
    # One context for the whole run.  Matching always targets the *animal's*
    # contours, never the collage growing on top of them, so round 4 is still
    # fitting objects to the duck rather than to round 3's paste.
    ctx = TargetContext(animal_outline, subject_mask=animal_subject)
    img_scale = proc.shape[1] / canvas.shape[1]

    for round_idx in range(rounds):
        t0 = time.time()
        print(f"\n{'=' * 60}\nROUND {round_idx + 1}/{rounds}\n{'=' * 60}")

        ctx.update_occupied(occupied_full)

        def score_one(idx, rank_only):
            if idx in used:
                return None
            r = match_variants(ctx, variant_cache[idx], config,
                               overlap_penalty=config["overlap_penalty"],
                               min_body_frac=config["min_body_frac"],
                               max_overlap=config["max_overlap"],
                               rank_only=rank_only)
            return None if r is None else (r, idx)

        # Two-stage cascade.  Every object is scored against the same immutable
        # context, so the sweep is embarrassingly parallel; and since ranking an
        # object costs ~15x less than matching it properly, the pool is first
        # ranked coarsely and only the most promising handful is re-matched at
        # full resolution.  Together these turn a linear scan of the whole pool
        # into something that scales to thousands of objects.
        def sweep(indices, rank_only):
            out = []
            with ThreadPoolExecutor(max_workers=threads) as ex:
                futures = [ex.submit(score_one, i, rank_only) for i in indices]
                for f in futures:
                    got = f.result()
                    if got is not None:
                        out.append(got)
            out.sort(key=lambda c: c[0]["score"], reverse=True)
            return out

        pending = [i for i in range(len(objects)) if i not in used]
        ranked = sweep(pending, rank_only=True)
        shortlist = [idx for _, idx in ranked[:config["rank_top_k"]]]
        candidates = sweep(shortlist, rank_only=False) if shortlist else []

        if not candidates:
            print("  no placement found; stopping early")
            break

        # Prefer a category not used yet.  Without this a pool holding many
        # images of one class tends to win every round with near-identical
        # objects, and the point is a composition of *different* things.
        pick = candidates[0]
        if not allow_repeat_class:
            fresh = next((c for c in candidates
                          if objects[c[1]]["class_name"] not in used_classes),
                         None)
            if fresh is not None:
                pick = fresh
        result, idx = pick
        obj = objects[idx]

        if result["score"] < config["min_score"]:
            print(f"  best remaining fit scores {result['score']:.3f}, below "
                  f"the {config['min_score']} floor -- stopping rather than "
                  f"pasting an object that does not fit")
            break

        used.add(idx)
        used_classes.add(obj["class_name"])

        result = dict(result)
        result["x"] = int(round(result["x"] / img_scale))
        result["y"] = int(round(result["y"] / img_scale))
        result["scale"] = result["match_scale"] * (obj["img_scale"] / img_scale)

        canvas, placed_mask = place_object(
            canvas, obj, result, blend_mode, alpha,
            clip_mask=animal_subject_full if clip_to_animal else None)
        occupied_full = np.maximum(occupied_full, placed_mask)
        history.append(canvas.copy())
        placed.append({
            "class_name": obj["class_name"],
            "source": obj["source"],
            "score": result["score"],
            "transform": result["transform"],
        })

        print(f"  + {obj['class_name']} ({obj['source']}) "
              f"score={result['score']:.3f} transform={result['transform']} "
              f"scale={result['scale']:.2f}  [{time.time() - t0:.1f}s]")

        if save_steps:
            os.makedirs(output_dir, exist_ok=True)
            step_path = os.path.join(output_dir,
                                     f"{animal_name}_round{round_idx + 1}.png")
            cv2.imwrite(step_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, f"{animal_name}_composed.png")
    cv2.imwrite(final_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"\nFinal image: {final_path}")

    return canvas, history, placed


def save_strip(history, placed, path):
    """Save a left-to-right strip of the composition, one frame per round."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(history)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.6))
    axes = np.atleast_1d(axes)
    for i, (ax, frame) in enumerate(zip(axes, history)):
        ax.imshow(frame)
        ax.axis("off")
        if i == 0:
            ax.set_title("original animal", fontsize=10, weight="bold")
        else:
            p = placed[i - 1]
            ax.set_title(f"+ {p['class_name']}\nscore={p['score']:.3f}",
                         fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Progress strip: {path}")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--animal", default="image_patches/duck3.png",
                   help="photo of the animal to build")
    p.add_argument("--rounds", type=int, default=4,
                   help="how many objects to lay onto the animal")
    p.add_argument("--sources", default="caltech101",
                   help="comma-separated: " + ",".join(sources.ARCHIVE_SOURCES)
                        + ",imagenet,folder")
    p.add_argument("--num-objects", type=int, default=150,
                   help="size of the object pool to search")
    p.add_argument("--folder", default=None,
                   help="directory for the 'folder' source")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--max-resolution", type=int, default=320)
    p.add_argument("--min-scale", type=float, default=0.25)
    p.add_argument("--max-scale", type=float, default=0.75)
    p.add_argument("--scale-steps", type=int, default=16)
    p.add_argument("--rotation-steps", type=int, default=8)
    p.add_argument("--no-rotation", action="store_true")
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--overlap-penalty", type=float, default=0.35,
                   help="discourage stacking objects on the same spot")
    p.add_argument("--min-body-frac", type=float, default=0.10,
                   help="each object must cover at least this fraction of the "
                        "animal's silhouette")
    p.add_argument("--blend", default="alpha", choices=["soft", "alpha", "replace"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--min-score", type=float, default=0.22,
                   help="stop early rather than place a poorly fitting object")
    p.add_argument("--rank-top-k", type=int, default=15,
                   help="objects promoted from the coarse ranking pass to "
                        "full-resolution matching each round")
    p.add_argument("--max-overlap", type=float, default=0.35,
                   help="reject a placement burying more than this fraction of "
                        "itself in already-placed objects")
    p.add_argument("--allow-repeat-class", action="store_true",
                   help="let the same object category win more than once")
    p.add_argument("--no-clip", action="store_true",
                   help="let objects spill outside the animal's silhouette")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-class-cap", type=int, default=3,
                   help="max images per category, for pool variety")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])

    if not segment.REMBG_AVAILABLE:
        raise SystemExit("rembg is required: pip install 'rembg[cpu]'")
    if not os.path.exists(args.animal):
        raise SystemExit(f"Animal image not found: {args.animal}")

    config = {
        "min_scale": args.min_scale,
        "max_scale": args.max_scale,
        "scale_steps": args.scale_steps,
        "max_resolution": args.max_resolution,
        "allow_flip": not args.no_flip,
        "allow_rotation": not args.no_rotation,
        "rotation_steps": args.rotation_steps,
        "overlap_penalty": args.overlap_penalty,
        "min_body_frac": args.min_body_frac,
        "min_score": args.min_score,
        "max_overlap": args.max_overlap,
        "rank_top_k": args.rank_top_k,
        "rank_coarse_factor": 6,
        "rank_scale_steps": 8,
        "rank_rotation_stride": 2,
        "coarse_factor": 3,
        "refine_top": 3,
        "early_stop_threshold": 1.1,
    }

    print(f"Animal:  {args.animal}")
    print(f"Sources: {args.sources}")
    print(f"Pool:    {args.num_objects} objects, {args.rounds} rounds\n")

    t0 = time.time()
    pool = sources.load_objects(
        args.sources.split(","), args.num_objects, seed=args.seed,
        per_class_cap=args.per_class_cap, folder=args.folder,
    )
    if not pool:
        raise SystemExit("No object images loaded. Check --sources / network.")
    print(f"Loaded {len(pool)} object images in {time.time() - t0:.1f}s\n")

    t0 = time.time()
    print("Segmenting object pool...")
    objects = prepare_objects(pool, args.max_resolution, num_threads=args.threads)
    print(f"{len(objects)} usable objects in {time.time() - t0:.1f}s")

    if not objects:
        raise SystemExit("No usable object silhouettes.")

    canvas, history, placed = compose(
        args.animal, objects, config, rounds=args.rounds,
        blend_mode=args.blend, alpha=args.alpha, output_dir=args.output_dir,
        threads=args.threads, clip_to_animal=not args.no_clip,
        allow_repeat_class=args.allow_repeat_class,
    )

    animal_name = os.path.splitext(os.path.basename(args.animal))[0]
    save_strip(history, placed,
               os.path.join(args.output_dir, f"{animal_name}_strip.png"))

    print("\nComposition:")
    for i, p in enumerate(placed, 1):
        print(f"  {i}. {p['class_name']:<28} {p['source']:<12} "
              f"score={p['score']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
