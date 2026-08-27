#!/usr/bin/env python3
"""Search Caltech101 for images that a patch composites onto.

Preset over search_targets.py.

    python autosearchcaltch.py
"""

import sys

from search_targets import main

if __name__ == "__main__":
    raise SystemExit(main(["--source", "caltech101"] + sys.argv[1:]))
