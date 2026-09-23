"""PyInstaller entry point for the Labelme package."""

import os

# PyInstaller can bundle third-party metadata with a non-UTF-8 encoding.
# Pydantic's plugin discovery reads every bundled entry point at import time,
# so disable optional plugins before importing OSAM/Labelme.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "__all__")

from labelme.__main__ import main


if __name__ == "__main__":
    main()
