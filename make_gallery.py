#!/usr/bin/env python3
"""Compose many canvases against one shared object pool.

Running compose_animal.py N times re-loads and re-segments the object pool N
times, which is the slow half of a run.  This does it once and reuses the pool
-- and the flip/rotate variants built from it -- across every canvas, so the
marginal cost of another picture is just its own rounds.

    python make_gallery.py --count 12
    python make_gallery.py --canvas a.jpg --canvas b.jpg --rounds 3

With no --canvas, canvases are auto-picked from a dataset by silhouette
quality, so the gallery is not full of pictures the segmenter could not cut.
"""

import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np

from compose_animal import compose, describe_silhouette
from composer import segment, sources
from composer.geometry import patch_variants, resize_to_max
from composer.pipeline import prepare_objects

# Categories with one clear subject and a strong outline, which is what makes a
# canvas worth composing onto.  Drawn from Caltech101's category names.
GOOD_CANVAS_CATEGORIES = [
    "airplanes", "anchor", "bonsai", "brontosaurus", "buddha", "butterfly",
    "cellphone", "chair", "chandelier", "cup", "dolphin", "dragonfly",
    "electric_guitar", "elephant", "emu", "ewer", "flamingo", "grand_piano",
    "hawksbill", "helicopter", "ibis", "kangaroo", "ketch", "lamp", "laptop",
    "llama", "lobster", "menorah", "Motorbikes", "octopus", "pagoda", "panda",
    "pyramid", "revolver", "rooster", "sax", "schooner", "scissors", "scorpion",
    "sea_horse", "starfish", "stegosaurus", "sunflower", "umbrella", "watch",
    "wheelchair", "windsor_chair", "yin_yang",
]


def silhouette_quality(img_rgb, max_resolution):
    """Score how cleanly an image's subject separates from its background."""
    proc, _ = resize_to_max(img_rgb, max_resolution)
    raw = segment.foreground_mask(proc)
    cleaned = segment.clean_mask(raw)
    subject = segment.largest_component(cleaned)

    coverage = np.count_nonzero(subject) / subject.size
    total = float(np.count_nonzero(cleaned))
    dominance = (np.count_nonzero(subject) / total) if total else 0.0

    contours, _ = cv2.findContours(subject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    biggest = max(contours, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(biggest))
    solidity = (cv2.contourArea(biggest) / hull) if hull else 0.0

    # A canvas wants a subject that fills a good part of the frame, is the only
    # thing in it, and has an interesting (not blob-like) outline.
    if not (0.12 <= coverage <= 0.75) or dominance < 0.75:
        return 0.0
    return dominance * (1.0 - abs(solidity - 0.6))


def pick_canvases(count, max_resolution, seed=0, scan=6):
    """Choose `count` canvases from Caltech101 by silhouette quality."""
    root = sources.ensure_archive("caltech101")
    rng = random.Random(seed)
    categories = [c for c in GOOD_CANVAS_CATEGORIES
                  if os.path.isdir(os.path.join(root, c))]
    rng.shuffle(categories)

    print(f"Scanning {len(categories)} categories for clean silhouettes...")
    chosen = []
    for category in categories:
        if len(chosen) >= count:
            break
        files = sorted(os.listdir(os.path.join(root, category)))
        rng.shuffle(files)
        best = None
        for fname in files[:scan]:
            path = os.path.join(root, category, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None or min(img.shape[:2]) < 120:
                continue
            score = silhouette_quality(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                       max_resolution)
            if score > 0 and (best is None or score > best[0]):
                best = (score, path)
        if best:
            print(f"  {category:<18} quality={best[0]:.2f}")
            chosen.append((best[1], category))
    return chosen


def contact_sheet(entries, path, columns=4, cell=460):
    """Lay the finished compositions out as one sheet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(entries) / columns))
    fig, axes = plt.subplots(rows, columns,
                             figsize=(3.4 * columns, 3.8 * rows), squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")

    for i, (name, image, placed, _hist) in enumerate(entries):
        ax = axes[i // columns][i % columns]
        ax.imshow(image)
        parts = " + ".join(p["class_name"] for p in placed)
        ax.set_title(f"{name}\n{parts}", fontsize=7.5)
        ax.axis("off")

    plt.suptitle("Compositions", fontsize=15, weight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"Contact sheet: {path}")


def write_steps(history, output_dir, name):
    """Save each round's frame, so the build order survives the run."""
    steps_dir = os.path.join(output_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)
    paths = []
    for i, frame in enumerate(history):
        path = os.path.join(steps_dir, f"{name}_{i:02d}.png")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


def step_sheets(entries, path_stem, rows_per_sheet=6):
    """One row per composition, one column per object as it lands.

    The finished picture hides the order things arrived in; this shows the
    canvas, then each object added on top of the last, which is the only view
    that makes a placement's contribution legible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    chunks = [entries[i:i + rows_per_sheet]
              for i in range(0, len(entries), rows_per_sheet)]

    for sheet_no, chunk in enumerate(chunks, 1):
        cols = max(len(h) for _n, _i, _p, h in chunk)
        rows = len(chunk)
        fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 3.25 * rows),
                                 squeeze=False)
        for ax in axes.flatten():
            ax.axis("off")

        for r, (name, _img, placed, history) in enumerate(chunk):
            for c, frame in enumerate(history):
                ax = axes[r][c]
                ax.imshow(frame)
                ax.axis("off")
                if c == 0:
                    ax.set_title(f"{name.replace('_', ' ')}\n(original)",
                                 fontsize=8, weight="bold")
                else:
                    p = placed[c - 1]
                    ax.set_title(f"+ {p['class_name']}\n{p['source']}  "
                                 f"{p['score']:.2f}", fontsize=8)

        plt.suptitle(f"Each object as it lands — sheet {sheet_no} of {len(chunks)}",
                     fontsize=14, weight="bold")
        plt.tight_layout()
        out = f"{path_stem}_{sheet_no:02d}.png"
        plt.savefig(out, dpi=118, bbox_inches="tight")
        plt.close(fig)
        print(f"Step sheet: {out}")
        written.append(out)
    return written


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--canvas", action="append", default=[],
                   help="image to compose onto; repeatable")
    p.add_argument("--count", type=int, default=12,
                   help="how many canvases to auto-pick when none are given")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--sources", default="caltech101,flowers102,imagenette")
    p.add_argument("--num-objects", type=int, default=500)
    p.add_argument("--per-class-cap", type=int, default=4)
    p.add_argument("--output-dir", default="gallery")
    p.add_argument("--max-resolution", type=int, default=320)
    p.add_argument("--min-scale", type=float, default=0.25)
    p.add_argument("--max-scale", type=float, default=0.75)
    p.add_argument("--scale-steps", type=int, default=16)
    p.add_argument("--rotation-steps", type=int, default=8)
    p.add_argument("--min-body-frac", type=float, default=0.10)
    p.add_argument("--max-overlap", type=float, default=0.35)
    p.add_argument("--min-containment", type=float, default=0.85)
    p.add_argument("--overlap-penalty", type=float, default=0.35)
    p.add_argument("--min-score", type=float, default=0.22)
    p.add_argument("--rank-top-k", type=int, default=15)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--columns", type=int, default=4)
    p.add_argument("--rows-per-sheet", type=int, default=6,
                   help="compositions per step sheet")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if not segment.REMBG_AVAILABLE:
        raise SystemExit("rembg is required: pip install 'rembg[cpu]'")

    config = {
        "min_scale": args.min_scale, "max_scale": args.max_scale,
        "scale_steps": args.scale_steps, "max_resolution": args.max_resolution,
        "allow_flip": True, "allow_rotation": True,
        "rotation_steps": args.rotation_steps,
        "overlap_penalty": args.overlap_penalty,
        "min_body_frac": args.min_body_frac,
        "min_score": args.min_score,
        "max_overlap": args.max_overlap,
        "min_containment": args.min_containment,
        "rank_top_k": args.rank_top_k,
        "coarse_factor": 3, "refine_top": 3,
        "rank_coarse_factor": 6, "rank_scale_steps": 8,
        "rank_rotation_stride": 2,
        "early_stop_threshold": 1.1,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    canvases = [(c, os.path.splitext(os.path.basename(c))[0]) for c in args.canvas]
    if not canvases:
        canvases = pick_canvases(args.count, args.max_resolution, args.seed)
    if not canvases:
        raise SystemExit("No canvases to compose onto.")
    print(f"\n{len(canvases)} canvases\n")

    t0 = time.time()
    pool = sources.load_objects(args.sources.split(","), args.num_objects,
                                seed=args.seed, per_class_cap=args.per_class_cap)
    if not pool:
        raise SystemExit("No object images loaded. Check --sources / network.")
    print(f"\nSegmenting {len(pool)} objects...")
    objects = prepare_objects(pool, args.max_resolution, num_threads=args.threads)
    print(f"{len(objects)} usable objects in {time.time() - t0:.1f}s")
    if not objects:
        raise SystemExit("No usable object silhouettes.")

    # Built once and shared by every canvas; this is the whole point of the
    # script.
    print("Building variants...")
    variants = [patch_variants(o["proc"], o["outline"], o["body"], True, True,
                               args.rotation_steps) for o in objects]
    print(f"  {sum(len(v) for v in variants)} variants\n")

    entries = []
    manifest = []
    for i, (path, label) in enumerate(canvases, 1):
        # Unique per composition even when two canvases share a filename.
        name = f"{label}_{os.path.splitext(os.path.basename(path))[0]}"
        name = name.replace(os.sep, "_").replace(" ", "_")
        print(f"\n{'#' * 60}\n# [{i}/{len(canvases)}] {label}\n{'#' * 60}")
        try:
            image, history, placed = compose(
                path, objects, config, rounds=args.rounds,
                output_dir=args.output_dir, save_steps=False,
                threads=args.threads, variant_cache=variants, name=name)
            step_paths = write_steps(history, args.output_dir, name)
        except Exception as exc:
            print(f"  ! failed: {type(exc).__name__}: {exc}")
            continue
        if placed:
            entries.append((label, image, placed, history))
            manifest.append({
                "name": name,
                "label": label,
                "canvas": os.path.abspath(path),
                "composed": os.path.join(args.output_dir, f"{name}_composed.png"),
                "steps": step_paths,
                "placed": [{"class_name": p["class_name"], "source": p["source"],
                            "score": round(float(p["score"]), 4)} for p in placed],
            })

    if not entries:
        raise SystemExit("Nothing composed.")

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    contact_sheet(entries, os.path.join(args.output_dir, "contact_sheet.png"),
                  columns=args.columns)
    step_sheets(entries, os.path.join(args.output_dir, "step_sheet"),
                rows_per_sheet=args.rows_per_sheet)

    print(f"\n{len(entries)} compositions in {args.output_dir}/")
    for name, _img, placed, _hist in entries:
        parts = ", ".join(f"{p['class_name']} ({p['score']:.2f})" for p in placed)
        print(f"  {name:<20} {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
