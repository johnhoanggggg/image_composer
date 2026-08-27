"""Pluggable object-image sources.

Every source yields dicts of {'image': HxWx3 uint8 RGB, 'class_name': str}.

Selection criterion for the bundled sets: one dominant subject, shot against a
plain or shallow-depth background, so rembg produces a clean silhouette.  That
is what makes a source usable here -- a cluttered multi-object scene segments
into an unusable blob.

  caltech101   9k photos, 101 object categories, mostly centred on plain
               backgrounds.  The best all-round source for this pipeline.
  flowers102   8k single flowers, very clean separation.
  pets         7k cat/dog portraits, one animal per frame.
  cars         16k cars, single subject, strong silhouette.
  birds        12k birds (CUB-200), single subject.
  imagenette   13k photos across 10 easily-separable ImageNet classes.
  imagewoof    13k dog photos across 10 breeds.
  figures      1k rendered human and horse figures on plain white -- the
               easiest silhouettes available, and unusually articulated ones.
  hands        2.5k rendered hands on plain white; long thin shapes that little
               else in the pool provides.
  imagenet     ImageNet-1k streamed from HuggingFace (needs network access to
               huggingface.co; not reachable from every sandbox).
  folder       Any local directory of images.

Pool size matters as much as pool variety: the matcher takes the best fit it
can find, so more candidates means closer fits.
"""

import os
import random
import shutil
import tarfile
import zipfile

import cv2
import numpy as np

from .cache import DATASET_DIR

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

FASTAI_BASE = "https://s3.amazonaws.com/fast-ai-imageclas"
TFDATA_BASE = "https://storage.googleapis.com/download.tensorflow.org/data"

# name -> (url, directory that appears after extraction, approx MB)
ARCHIVE_SOURCES = {
    "caltech101": (f"{FASTAI_BASE}/caltech_101.tgz", "101_ObjectCategories", 126),
    "flowers102": (f"{FASTAI_BASE}/oxford-102-flowers.tgz", "oxford-102-flowers", 329),
    "pets": (f"{FASTAI_BASE}/oxford-iiit-pet.tgz", "oxford-iiit-pet", 774),
    "cars": (f"{FASTAI_BASE}/stanford-cars.tgz", "stanford-cars", 1867),
    "birds": (f"{FASTAI_BASE}/CUB_200_2011.tgz", "CUB_200_2011", 1097),
    "imagenette": (f"{FASTAI_BASE}/imagenette2-320.tgz", "imagenette2-320", 326),
    "imagewoof": (f"{FASTAI_BASE}/imagewoof2-320.tgz", "imagewoof2-320", 328),
    "figures": (f"{TFDATA_BASE}/horse-or-human.zip", "horse-or-human", 142),
    "hands": (f"{TFDATA_BASE}/rps.zip", "rps", 191),
}

# Caltech101 categories that are background clutter or too texture-like to
# give a usable silhouette.
CALTECH_SKIP = {"BACKGROUND_Google", "Faces", "Faces_easy"}

# Imagenette/imagewoof name their class folders with WordNet ids.  Mapping them
# back keeps run output readable ("French horn" rather than "n03394916").
WORDNET_NAMES = {
    "n01440764": "tench", "n02102040": "English springer",
    "n02979186": "cassette player", "n03000684": "chain saw",
    "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump",
    "n03445777": "golf ball", "n03888257": "parachute",
    "n02086240": "Shih-Tzu", "n02087394": "Rhodesian ridgeback",
    "n02088364": "beagle", "n02089973": "English foxhound",
    "n02093754": "Border terrier", "n02096294": "Australian terrier",
    "n02099601": "golden retriever", "n02105641": "Old English sheepdog",
    "n02111889": "Samoyed", "n02115641": "dingo",
}

# Sources whose images sit in one flat directory, with no class folder to read
# a name from.
FLAT_SOURCES = {"flowers102": "flower", "cars": "car"}

# Sources that are flat but encode the class in the filename, e.g. the Oxford
# pets set's "great_pyrenees_133.jpg".
FILENAME_LABEL_SOURCES = {"pets"}


def _download(url, dest):
    import urllib.request

    print(f"  downloading {os.path.basename(dest)} ...")
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"    {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="\r")
    os.replace(tmp, dest)
    print()


def ensure_archive(name, keep_archive=False):
    """Download and extract an archive source, returning its root directory.

    The extracted tree is the cache, so the archive is removed once it has been
    unpacked -- keeping both roughly doubles the disk this needs, and several of
    these sets run to a couple of gigabytes.
    """
    if name not in ARCHIVE_SOURCES:
        raise ValueError(f"Unknown archive source: {name}")
    url, dirname, size_mb = ARCHIVE_SOURCES[name]
    os.makedirs(DATASET_DIR, exist_ok=True)
    root = os.path.join(DATASET_DIR, dirname)
    if os.path.isdir(root):
        return root

    archive = os.path.basename(url)
    archive_path = os.path.join(DATASET_DIR, archive)
    if not os.path.exists(archive_path):
        print(f"Fetching '{name}' (~{size_mb} MB, one time)")
        _download(url, archive_path)

    print(f"  extracting {archive} ...")
    # Some of these archives unpack loose into the current directory rather
    # than into a folder of their own, so extract into a staging directory and
    # move the result into place.
    staging = os.path.join(DATASET_DIR, f".{dirname}.staging")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(staging)
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(staging)

    entries = os.listdir(staging)
    if len(entries) == 1 and os.path.isdir(os.path.join(staging, entries[0])):
        shutil.move(os.path.join(staging, entries[0]), root)
        shutil.rmtree(staging, ignore_errors=True)
    else:
        shutil.move(staging, root)

    if not keep_archive:
        try:
            os.remove(archive_path)
        except OSError:
            pass
    return root


# Directories inside a source that hold annotations, not photographs.
SKIP_DIRS = {
    "caltech101": set(CALTECH_SKIP),
    "birds": {"parts", "attributes", "segmentations"},
    "pets": {"annotations"},
}


def _list_images(root, skip_dirs=()):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTENSIONS:
                files.append(os.path.join(dirpath, fn))
    return files


def _read_rgb(path, min_side=64):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if min(img.shape[:2]) < min_side:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _class_name_from_path(path, root, source=None):
    if source in FLAT_SOURCES:
        return FLAT_SOURCES[source]

    if source in FILENAME_LABEL_SOURCES:
        stem = os.path.splitext(os.path.basename(path))[0]
        # "great_pyrenees_133" -> "great pyrenees"
        parts = stem.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            parts = parts[:-1]
        return " ".join(parts).lower()

    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if len(parts) >= 2:
        name = parts[-2]
        if name in WORDNET_NAMES:
            return WORDNET_NAMES[name]
        # Strip dataset-specific numeric prefixes, e.g. "001.Black_footed_Albatross"
        if "." in name and name.split(".")[0].isdigit():
            name = name.split(".", 1)[1]
        return name.replace("_", " ")
    return os.path.splitext(parts[-1])[0]


def load_archive_source(name, num_images, rng, per_class_cap=None):
    root = ensure_archive(name)
    files = _list_images(root, skip_dirs=SKIP_DIRS.get(name, ()))
    if not files:
        return []

    rng.shuffle(files)

    # A flat source has one class name for every image, so capping per class
    # would admit exactly one image from it.
    if per_class_cap and name not in FLAT_SOURCES:
        by_class = {}
        for f in files:
            by_class.setdefault(_class_name_from_path(f, root, name), []).append(f)

        # Raise the cap when a source has too few categories to fill its share.
        # A fixed cap of 3 lets a 200-category set contribute 600 images but a
        # 3-category set only 9, quietly starving exactly the sources whose
        # silhouettes are cleanest.
        needed = int(np.ceil(num_images / max(1, len(by_class))))
        cap = max(per_class_cap, needed)

        kept = []
        for members in by_class.values():
            kept.extend(members[:cap])
        rng.shuffle(kept)
        files = kept

    out = []
    for path in files:
        if len(out) >= num_images:
            break
        img = _read_rgb(path)
        if img is None:
            continue
        out.append({"image": img,
                    "class_name": _class_name_from_path(path, root, name)})
    return out


def load_folder_source(folder, num_images, rng):
    files = _list_images(folder)
    if not files:
        return []
    rng.shuffle(files)
    out = []
    for path in files[: num_images * 2]:
        if len(out) >= num_images:
            break
        img = _read_rgb(path)
        if img is None:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        out.append({"image": img, "class_name": name})
    return out


def load_imagenet_source(num_images, rng, split="validation"):
    """Stream ImageNet-1k from HuggingFace.

    Requires reachable huggingface.co; raises otherwise so the caller can fall
    back to another source.
    """
    from datasets import load_dataset

    ds = load_dataset("imagenet-1k", split=split, streaming=True,
                      trust_remote_code=True)
    it = iter(ds)
    first = next(it)

    image_key = next((k for k in ("image", "jpg", "png", "webp") if k in first), None)
    label_key = next((k for k in ("label", "cls", "class", "target") if k in first), None)
    if image_key is None or label_key is None:
        raise RuntimeError(f"Unexpected ImageNet schema: {list(first.keys())}")

    names = None
    try:
        names = ds.features[label_key].names
    except Exception:
        pass

    def extract(item):
        img = item[image_key]
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = item[label_key]
        return {
            "image": np.array(img),
            "class_name": names[label] if names else f"class_{label}",
        }

    # Reservoir sample so the selection is not just the head of the stream.
    samples = [extract(first)]
    for i, item in enumerate(it, start=1):
        if len(samples) < num_images:
            samples.append(extract(item))
        else:
            j = rng.randint(0, i)
            if j < num_images:
                samples[j] = extract(item)
        if i >= num_images * 10:
            break
    return samples


def load_objects(sources, num_images, seed=0, per_class_cap=None,
                 folder=None, imagenet_split="validation"):
    """Load `num_images` object photos, split evenly across `sources`.

    Unreachable sources are skipped with a warning rather than aborting the
    run, so a machine without HuggingFace access still gets a full pool.
    """
    rng = random.Random(seed)
    if isinstance(sources, str):
        sources = [sources]

    per_source = max(1, num_images // max(1, len(sources)))
    pool = []
    for name in sources:
        try:
            if name == "imagenet":
                got = load_imagenet_source(per_source, rng, imagenet_split)
            elif name == "folder":
                if not folder:
                    raise ValueError("source 'folder' requires folder=...")
                got = load_folder_source(folder, per_source, rng)
            else:
                got = load_archive_source(name, per_source, rng, per_class_cap)
        except Exception as exc:
            print(f"  ! source '{name}' unavailable ({type(exc).__name__}: {exc})")
            continue
        print(f"  source '{name}': {len(got)} images")
        for item in got:
            item["source"] = name
        pool.extend(got)

    rng.shuffle(pool)
    return pool[:num_images]
