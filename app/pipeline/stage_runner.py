"""
app/pipeline/stage_runner.py — Runs individual pipeline stages with error handling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StageRunner:
    """Represents a single executable pipeline stage."""

    name: str
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    critical: bool = True
    description: str = ""


def run_stages(
    stages: List[StageRunner],
    context: Dict[str, Any],
    progress_callback: Optional[Callable[[str, str, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run a list of stages sequentially, updating context along the way.

    Parameters
    ----------
    stages:
        Ordered list of StageRunner instances to execute.
    context:
        Dictionary shared across stages containing job parameters and intermediates.
    progress_callback:
        Optional callback invoked with (stage_name, status, message).
        Status values: 'running', 'done', 'failed', 'skipped'.

    Returns
    -------
    Dict[str, Any]
        Updated context dictionary.
    """
    for stage in stages:
        logger.info(f"Starting stage: {stage.name} - {stage.description}")
        if progress_callback:
            progress_callback(stage.name, "running", stage.description or f"Running {stage.name}...")

        start_time = time.time()
        try:
            result = stage.fn(context)
            if isinstance(result, dict):
                context.update(result)

            elapsed = time.time() - start_time
            msg = f"Completed in {elapsed:.1f}s"
            logger.info(f"Stage '{stage.name}' succeeded in {elapsed:.1f}s")
            if progress_callback:
                progress_callback(stage.name, "done", msg)

        except Exception as exc:
            elapsed = time.time() - start_time
            err_msg = f"Failed after {elapsed:.1f}s: {exc}"
            logger.error(f"Stage '{stage.name}' failed: {exc}", exc_info=True)
            if progress_callback:
                progress_callback(stage.name, "failed", err_msg)

            if stage.critical:
                raise RuntimeError(f"Critical stage '{stage.name}' failed: {exc}") from exc
            else:
                logger.warning(f"Non-critical stage '{stage.name}' failed, continuing pipeline.")

    return context
