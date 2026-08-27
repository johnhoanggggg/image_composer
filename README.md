# image_composer

Build an animal out of photographs of other objects.

![A duck composed of a hedgehog, a flower, a crab and a church](examples/duck3_strip.png)

The animal photo is the canvas. Each round, every object in a pool is scored
against the animal's contours and the best-fitting one is composited on top,
clipped to the animal's silhouette. After four rounds the duck above is a
hedgehog, a flower petal, a crab and a church — and still reads as a duck.

## Install

```bash
pip install -r requirements.txt
```

`rembg` downloads its ~180 MB segmentation model on first use.

## Compose an animal

```bash
python compose_animal.py --animal image_patches/duck3.png --rounds 4
```

Useful flags:

| flag | meaning |
| --- | --- |
| `--rounds N` | how many objects to lay onto the animal |
| `--sources a,b,c` | which image databases to draw objects from |
| `--num-objects N` | size of the object pool to search |
| `--min-body-frac F` | each object must cover this much of the animal (keeps objects legible) |
| `--max-overlap F` | reject a placement that buries this much of itself in earlier objects |
| `--min-score F` | stop early rather than paste something that does not fit |
| `--no-clip` | let objects spill outside the animal's silhouette |
| `--rank-top-k N` | objects promoted from coarse ranking to full-resolution matching |

## Search the other direction

`search_targets.py` takes patches from a folder and finds dataset images they
composite convincingly *onto* — the workflow that puts a duck's head on a
sailboat. Click any cell in the result grid to save that composite.

```bash
python search_targets.py --patch-dir image_patches --source caltech101
```

`autosearchcaltch.py` and `auto_search_imagenet.py` are presets over it, for
Caltech101 and ImageNet respectively.

## Image sources

Objects come from datasets of *single subjects on plain backgrounds*, because
that is what a background remover can cut cleanly; a cluttered scene segments
into an unusable blob. Archives download once into
`~/.cache/image_composer/datasets`.

| `--source` | contents | size |
| --- | --- | --- |
| `caltech101` | 9k photos, 101 object categories — best all-round source | 126 MB |
| `flowers102` | 8k single flowers, very clean separation | 329 MB |
| `imagenette` | 13k photos, 10 easily-separable ImageNet classes | 326 MB |
| `pets` | 7k cat and dog portraits | 774 MB |
| `birds` | 12k birds (CUB-200) | 1.1 GB |
| `cars` | 16k cars, strong silhouettes | 1.9 GB |
| `imagenet` | ImageNet-1k streamed from HuggingFace | — |
| `folder` | any local directory, via `--folder` | — |

`imagenet` needs network access to `huggingface.co`; the rest come from S3
mirrors. Unreachable sources are skipped with a warning rather than aborting
the run, so a machine behind a restrictive proxy still gets a full pool.

## How the matching works

Placements are scored by a symmetric chamfer match between the object's
silhouette and the animal's contours, combining three terms:

- **precision** — do the object's edges land on the animal's edges?
- **recall** — how much of the animal's whole contour does this placement
  explain? Without this, the score is maximised by shrinking an object to a
  few pixels, which sits perfectly on any contour.
- **containment** — how much of the object's body lands inside the animal?

Objects already placed are recorded in an occupancy mask, which both penalises
and hard-limits overlap, so later rounds spread across the animal instead of
burying earlier ones.

A run searches the pool in two stages: every object is ranked at coarse
resolution, and only the best `--rank-top-k` are re-matched at full
resolution. Ranking an object is ~15× cheaper than matching it, so the pool
can grow without the round time following it.

## Layout

```
composer/
  segment.py      background removal, outlines, mask cache
  geometry.py     resizing, flip/rotate variants, transform replay
  matching.py     chamfer scoring, coarse-to-fine search
  compositing.py  pasting a cut-out onto a canvas
  sources.py      pluggable image databases
  pipeline.py     segmenting an object pool
  search.py       the patch-to-target direction
  cache.py        on-disk mask cache
compose_animal.py   build an animal out of objects
search_targets.py   find images a patch composites onto
```
