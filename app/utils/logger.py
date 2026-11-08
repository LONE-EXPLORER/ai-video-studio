"""
app/utils/logger.py — Structured logging setup.

Usage:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from typing import Optional


class _RedactingFilter(logging.Filter):
    """Filters out records that contain API key patterns."""

    _SENSITIVE_KEYWORDS = ("api_key", "apikey", "secret", "password", "token")

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        message = record.getMessage().lower()
        for kw in self._SENSITIVE_KEYWORDS:
            if kw in message:
                # Redact the record message in-place
                record.msg = "[REDACTED — sensitive data omitted from logs]"
                record.args = ()
                break
        return True


def _get_log_level() -> int:
    """Read log level from config without creating a circular import."""
    try:
        from app.config import get_config  # noqa: PLC0415

        level_name = get_config().log_level
    except Exception:  # pragma: no cover
        level_name = "INFO"
    return getattr(logging, level_name, logging.INFO)


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with a stdout StreamHandler and timestamp formatting.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — return as-is (handles repeated imports)
        return logger

    logger.setLevel(_get_log_level())

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(_get_log_level())

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.addFilter(_RedactingFilter())

    logger.addHandler(handler)
    logger.propagate = False  # Don't duplicate to root logger

    return logger
