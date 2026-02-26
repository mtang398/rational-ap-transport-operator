"""Logging setup utilities."""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
):
    """Configure root logger."""
    if format_str is None:
        format_str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_str,
        handlers=handlers,
    )
