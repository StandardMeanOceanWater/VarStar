# -*- coding: utf-8 -*-
"""
Legacy compatibility entry point for the Bayer split stage.

The canonical owner is now `split.py`. This module keeps the transitional
`split_bayer.py` import path and CLI entry working during the rename period.
"""

from __future__ import annotations

from pathlib import Path

from split import (
    _BAYER_PATTERNS,
    _build_channel_header,
    _propagate_wcs,
    detect_bayer_pattern,
    run_debayer,
    run_split,
)

__all__ = [
    "_BAYER_PATTERNS",
    "_build_channel_header",
    "_propagate_wcs",
    "detect_bayer_pattern",
    "run_debayer",
    "run_split",
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        cfg_path = Path(__file__).parent / "observation_config.yaml"
    else:
        cfg_path = Path(sys.argv[1])

    print("[WARN] split_bayer.py is a legacy entry point. Use split.py or run_pipeline.py --step split.")
    run_split(cfg_path)
