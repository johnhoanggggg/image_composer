"""The other direction: given patches, find dataset images that fit them.

`compose_animal` treats the animal as the canvas.  This module is the original
workflow -- a patch (a duck's head, say) is matched against a pool of dataset
photos to find the ones whose subject it lands on convincingly, and the top
matches are shown as a grid to pick from.
"""

import os

import cv2
import numpy as np
from tqdm import tqdm

from . import segment, sources
from .compositing import composite
from .geometry import apply_transform, crop_to_content, patch_variants, resize_to_max
from .matching import TargetContext, match_targets
from .pipeline import prepare_object


def load_patches(patch_dir, max_resolution):
    """Segment every image in a folder into a matchable patch."""
    if not os.path.isdir(patch_dir):
        raise SystemExit(f"Patch folder not found: {patch_dir}")

    files = sorted(f for f in os.listdir(patch_dir)
                   if os.path.splitext(f)[1].lower() in sources.IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"No images in {patch_dir}")

    patches = []
    for i, fname in enumerate(files, 1):
        img = cv2.imread(os.path.join(patch_dir, fname), cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 64:
            print(f"  [{i}/{len(files)}] {fname} - skipped")
            continue
        item = {
            "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            "class_name": os.path.splitext(fname)[0],
            "source": "patch",
        }
        prepared = prepare_object(item, max_resolution)
        if prepared is None:
            print(f"  [{i}/{len(files)}] {fname} - no subject found")
            continue
        print(f"  [{i}/{len(files)}] {fname} - loaded")
        patches.append(prepared)
    return patches


def prepare_targets(pool, max_resolution, num_threads=4):
    """Turn dataset images into match targets with prebuilt contexts."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    segment.get_session()

    def work(item):
        img = item["image"]
        proc, img_scale = resize_to_max(img, max_resolution)
        mask = segment.foreground_mask(proc)
        outline = segment.mask_to_outline(mask)
        if np.count_nonzero(outline) < 60:
            return None
        return {
            "image": img,
            "img_scale": img_scale,
            "class_name": item.get("class_name", "target"),
            "ctx": TargetContext(outline, subject_mask=segment.clean_mask(mask)),
        }

    out = []
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = [ex.submit(work, i) for i in pool]
        for f in tqdm(as_completed(futures), total=len(futures)):
            r = f.result()
            if r is not None:
                out.append(r)
    return out


def search(patches, targets, config, top_k=25, num_threads=4,
           match_threshold=0.0):
    """Score every patch against every target, keeping the best `top_k`."""
    for i, patch in enumerate(patches, 1):
        print(f"\nSearching patch {i}/{len(patches)}: {patch['class_name']}")
        variants = patch_variants(
            patch["proc"], patch["outline"], patch["body"],
            config["allow_flip"], config["allow_rotation"],
            config["rotation_steps"])
        print(f"  {len(variants)} variants")

        cfg = dict(config)
        cfg["patch_scale"] = patch["img_scale"]
        cfg.setdefault("min_containment", 0.0)
        results = match_targets(targets, variants, cfg, num_threads=num_threads)

        results.sort(key=lambda r: r["score"], reverse=True)
        results = [r for r in results if r["score"] >= match_threshold]
        patch["matches"] = results[:top_k]

        if patch["matches"]:
            best = patch["matches"][0]
            print(f"  best: {best['class_name']} score={best['score']:.3f}")
        else:
            print(f"  no match above {match_threshold}")
    return patches


def build_composite(patch, match, blend_mode="alpha", alpha=1.0):
    """Render one patch/target pairing at full resolution."""
    transform = match["transform"]
    bx, by, bw, bh = patch["bbox_proc"]
    s = patch["img_scale"]
    fh, fw = patch["image"].shape[:2]
    x0, y0 = int(round(bx / s)), int(round(by / s))
    x1, y1 = min(fw, int(round((bx + bw) / s))), min(fh, int(round((by + bh) / s)))
    img = patch["image"][y0:y1, x0:x1]
    if img.size == 0:
        img = patch["image"]

    mask = cv2.resize(patch["body"], (img.shape[1], img.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    mask = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0

    img = apply_transform(img, transform)
    mask = apply_transform(mask, transform, cv2.INTER_LINEAR)

    rot = transform.split("_")[0]
    if rot.startswith("rot") and rot[3:] not in ("", "0", "90", "180", "270"):
        binary = (mask > 0.1).astype(np.uint8) * 255
        binary, img, mask = crop_to_content(binary, img, mask, reference=0)

    return composite(img, match["image"], match["x"], match["y"],
                     match["scale"], blend_mode, alpha, patch_mask=mask)


def visualize(patches, output_path, blend_mode="alpha", alpha=1.0,
              grid=None, interactive=True):
    """Grid of each patch's best matches; click a cell to save it.

    Saves the grid either way, so the tool is still useful over SSH or in a
    container with no display.
    """
    import matplotlib
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, ext = os.path.splitext(output_path)
    save_dir = os.path.dirname(output_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    saved = [0]

    for patch in patches:
        matches = patch.get("matches") or []
        if not matches:
            continue
        if grid:
            matches = matches[:grid * grid]
        # Size the grid to what there is to show; a fixed 5x5 left most of the
        # figure blank whenever fewer matches cleared the threshold.
        side = int(np.ceil(np.sqrt(len(matches))))
        rows = int(np.ceil(len(matches) / side))

        fig, axes = plt.subplots(rows, side, figsize=(3 * side, 3.3 * rows),
                                 squeeze=False)
        axes = np.atleast_1d(axes).flatten()
        cells = {}

        for j, match in enumerate(matches):
            ax = axes[j]
            img = build_composite(patch, match, blend_mode, alpha)
            ax.imshow(img)
            ax.set_title(f"#{j + 1}: {match['class_name']}\n"
                         f"score={match['score']:.3f} s={match['scale']:.2f}",
                         fontsize=8)
            ax.axis("off")
            cells[ax] = (img, patch["class_name"], match["class_name"])

        for j in range(len(matches), len(axes)):
            axes[j].axis("off")

        def on_click(event, cells=cells):
            if event.inaxes not in cells:
                return
            img, patch_name, match_name = cells[event.inaxes]
            saved[0] += 1
            path = os.path.join(save_dir,
                                f"saved_{patch_name}+{match_name}_{saved[0]:03d}.png")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Saved: {path}")
            for spine in event.inaxes.spines.values():
                spine.set_edgecolor("lime")
                spine.set_linewidth(4)
            event.inaxes.axis("on")
            event.inaxes.set_xticks([])
            event.inaxes.set_yticks([])
            event.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.suptitle(f"Top matches for: {patch['class_name']} "
                     f"(click a cell to save)", fontsize=12, weight="bold")
        plt.tight_layout()
        page = f"{base}_{patch['class_name']}{ext}"
        plt.savefig(page, dpi=130, bbox_inches="tight")
        print(f"Saved visualization: {page}")
        if interactive:
            plt.show()
        else:
            plt.close(fig)
