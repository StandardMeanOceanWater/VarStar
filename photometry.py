# -*- coding: utf-8 -*-
"""Step 4 photometry CLI wrapper."""

from __future__ import annotations

import os
import sys
import warnings
from functools import partial


VERSION = "1.79"

warnings.filterwarnings("ignore", message=".*datfix.*")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

if not os.environ.get("MPLBACKEND"):
    os.environ["MPLBACKEND"] = "Agg"

from pipeline_config import load_pipeline_config  # noqa: E402
from phot_config import Cfg, cfg_from_yaml  # noqa: E402
from phot_sources.photometry_backend import run_photometry_on_wcs_dir  # noqa: E402
from phot_sources.runner import run_main as _owner_run_main  # noqa: E402
from phot_sources.stage4 import (  # noqa: E402
    _run_stage4_postprocess as _owner_run_stage4_postprocess,
)
from phot_sources.vsx import (  # noqa: E402
    _prepare_vsx_candidates as _owner_prepare_vsx_candidates,
    _run_vsx_targets_for_target as _owner_run_vsx_targets_for_target,
)


__all__ = [
    "Cfg",
    "VERSION",
    "cfg_from_yaml",
    "load_pipeline_config",
    "main",
    "run_photometry_on_wcs_dir",
]


def main(argv=None):
    return _owner_run_main(
        argv,
        photometry_func=run_photometry_on_wcs_dir,
        stage4_postprocess_func=_owner_run_stage4_postprocess,
        prepare_vsx_candidates_func=_owner_prepare_vsx_candidates,
        run_vsx_targets_func=partial(
            _owner_run_vsx_targets_for_target,
            photometry_func=run_photometry_on_wcs_dir,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
