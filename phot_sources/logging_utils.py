"""Photometry logging and tracing helpers."""

from __future__ import annotations

import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable

DEFAULT_SUMMARY_PREFIXES = (
    "[sigma-clip]",
    "[CSV]",
    "[\u5b8c\u6210]",
    "[LS]",
    "[Fourier]",
    "[\u6bd4\u8f03\u661f]",
    "[\u751f\u9577\u66f2\u7dda]",
    "[SKIP]",
    "\u5b54\u5f91",
    "======",
    "  \u901a\u9053",
    "  FITS",
    "  \u8f38\u51fa",
    "\u6240\u6709\u901a\u9053",
    "[photometry]",
    "\u627e\u5230",
)


def emit_progress(logger: logging.Logger | None, message: str) -> None:
    progress_message = f"[progress] {message}"
    print(progress_message)
    if logger is not None:
        logger.info(progress_message)


def emit_progress_done(
    logger: logging.Logger | None,
    stage_label: str,
    started_at: float,
) -> None:
    emit_progress(logger, f"{stage_label} done in {time.perf_counter() - started_at:.1f}s")


def build_log_timestamp(
    out_tag: str | None = None,
    now: datetime | None = None,
) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    if out_tag:
        stamp = f"{stamp}_{out_tag}"
    return stamp


def detach_file_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


def attach_file_handler(
    logger: logging.Logger,
    log_path: Path,
    *,
    level: int = logging.DEBUG,
    encoding: str = "utf-8",
) -> logging.FileHandler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding=encoding)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return handler


def init_summary_logger(
    *,
    logger_name: str = "photometry",
    stream=None,
    keep_prefixes: Iterable[str] | None = None,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    prefixes = tuple(keep_prefixes or DEFAULT_SUMMARY_PREFIXES)
    stream_handler = logging.StreamHandler(stream or sys.stdout)
    stream_handler.setLevel(logging.INFO)

    class _SummaryFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            return record.levelno >= logging.ERROR or any(message.startswith(p) for p in prefixes)

    stream_handler.addFilter(_SummaryFilter())
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    return logger


def redirect_warnings_to_logger(logger: logging.Logger):
    previous_showwarning = warnings.showwarning
    warnings.showwarning = (
        lambda msg, cat, fn, ln, f=None, li=None: logger.debug("[astropy] %s", str(msg))
    )
    return previous_showwarning


__all__ = [
    "DEFAULT_SUMMARY_PREFIXES",
    "attach_file_handler",
    "build_log_timestamp",
    "detach_file_handlers",
    "emit_progress",
    "emit_progress_done",
    "init_summary_logger",
    "redirect_warnings_to_logger",
]
