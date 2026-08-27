#!/usr/bin/env python3
"""Find dataset images that a patch composites convincingly onto.

The reverse of compose_animal: here the patch is the subject and the search is
for backgrounds whose shape it fits -- the workflow that produced the duck
head on a couch.

    python search_targets.py --patch-dir image_patches --source caltech101
    python search_targets.py --patch-dir image_patches --source imagenet

Click any cell in the result grid to save that composite.
"""

import argparse
import os
import sys

from composer import search, segment, sources


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch-dir", default="image_patches")
    p.add_argument("--source", default="caltech101",
                   help="one of: " + ", ".join(sources.ARCHIVE_SOURCES)
                        + ", imagenet, folder")
    p.add_argument("--target-folder", default=None,
                   help="directory for --source folder")
    p.add_argument("--num-targets", type=int, default=200)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--match-threshold", type=float, default=0.2)
    p.add_argument("--output-dir", default="search_output")
    p.add_argument("--max-resolution", type=int, default=320)
    p.add_argument("--min-scale", type=float, default=0.3)
    p.add_argument("--max-scale", type=float, default=0.8)
    p.add_argument("--scale-steps", type=int, default=16)
    p.add_argument("--rotation-steps", type=int, default=8)
    p.add_argument("--no-rotation", action="store_true")
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--min-body-frac", type=float, default=0.05)
    p.add_argument("--blend", default="alpha", choices=["soft", "alpha", "replace"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--imagenet-split", default="validation")
    p.add_argument("--no-show", action="store_true",
                   help="save the grids without opening a window")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if not segment.REMBG_AVAILABLE:
        raise SystemExit("rembg is required: pip install 'rembg[cpu]'")

    config = {
        "min_scale": args.min_scale,
        "max_scale": args.max_scale,
        "scale_steps": args.scale_steps,
        "max_resolution": args.max_resolution,
        "allow_flip": not args.no_flip,
        "allow_rotation": not args.no_rotation,
        "rotation_steps": args.rotation_steps,
        "coarse_factor": 3,
        "refine_top": 3,
        "min_body_frac": args.min_body_frac,
        "early_stop_threshold": 1.1,
    }

    print(f"Patches: {args.patch_dir}")
    print(f"Source:  {args.source}  ({args.num_targets} targets)\n")

    patches = search.load_patches(args.patch_dir, args.max_resolution)
    print(f"\n{len(patches)} patches loaded\n")

    pool = sources.load_objects([args.source], args.num_targets, seed=args.seed,
                                folder=args.target_folder,
                                imagenet_split=args.imagenet_split)
    if not pool:
        raise SystemExit("No target images loaded. Check --source / network.")

    print(f"\nSegmenting {len(pool)} targets...")
    targets = search.prepare_targets(pool, args.max_resolution, args.threads)
    print(f"{len(targets)} usable targets\n")
    if not targets:
        raise SystemExit("No usable target silhouettes.")

    search.search(patches, targets, config, top_k=args.top_k,
                  num_threads=args.threads,
                  match_threshold=args.match_threshold)

    matched = [p for p in patches if p.get("matches")]
    if not matched:
        raise SystemExit(f"Nothing scored above {args.match_threshold}. "
                         "Lower --match-threshold or raise --num-targets.")

    os.makedirs(args.output_dir, exist_ok=True)
    search.visualize(matched, os.path.join(args.output_dir, "matches.png"),
                     blend_mode=args.blend, alpha=args.alpha,
                     interactive=not args.no_show)

    print("\nBest match per patch:")
    for p in matched:
        best = p["matches"][0]
        print(f"  {p['class_name']:<22} -> {best['class_name']:<24} "
              f"score={best['score']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
