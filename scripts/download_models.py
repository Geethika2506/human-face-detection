#!/usr/bin/env python3
"""Download the pinned YuNet and SFace ONNX files into models/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_assets import download_models  # noqa: E402


def main() -> None:
    download_models()


if __name__ == "__main__":
    main()
