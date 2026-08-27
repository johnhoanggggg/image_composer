#!/usr/bin/env python3
"""Search ImageNet for images that a patch composites onto.

Preset over search_targets.py.  Requires network access to huggingface.co;
where that is blocked, use autosearchcaltch.py or pass another --source.

    python auto_search_imagenet.py
"""

import sys

from search_targets import main

if __name__ == "__main__":
    raise SystemExit(main(["--source", "imagenet", "--imagenet-split", "train"]
                          + sys.argv[1:]))
