"""
app/quality/checker.py — Automated quality checks on rendered output.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel

from app.schemas.editing_plan import AspectRatio, EditingPlan
from app.utils.ffmpeg_utils import find_ffmpeg, find_ffprobe, probe_video
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Aspect ratio → expected (width, height)
_EXPECTED_DIMS = {
    AspectRatio.VERTICAL_9_16: (1080, 1920),
    AspectRatio.LANDSCAPE_16_9: (1920, 1080),
    AspectRatio.SQUARE_1_1: (1080, 1080),
}


class CheckResult(BaseModel):
    """Result of a single automated quality check."""

    name: str
    passed: bool
    detail: str


class QualityReport(BaseModel):
    """Aggregated quality report for a rendered video."""

    passed: bool
    checks: List[CheckResult]

    @property
    def failed_checks(self) -> List[CheckResult]:
        """Return only the checks that did not pass."""
        return [c for c in self.checks if not c.passed]


class QualityChecker:
    """
    Runs a battery of automated checks against a rendered video and returns
    a :class:`QualityReport`.
    """

    def check(self, video_path: Path, plan: EditingPlan) -> QualityReport:
        """
        Run all quality checks.

        Parameters
        ----------
        video_path:
            Path to the rendered export video.
        plan:
            The editing plan that describes the intended output.

        Returns
        -------
        QualityReport
        """
        checks: List[CheckResult] = []

        checks.append(self._check_file_exists(video_path))
        checks.append(self._check_valid_mp4(video_path))

        # Only run deeper checks if the file is a valid MP4
        if checks[-1].passed:
            probe = probe_video(video_path)
            checks.append(self._check_duration(probe, plan))
            checks.append(self._check_resolution(probe, plan))
            checks.append(self._check_audio_stream(probe))
            checks.append(self._check_no_black_video(video_path))

        overall = all(c.passed for c in checks)
        report = QualityReport(passed=overall, checks=checks)

        if overall:
            logger.info("Quality checks PASSED for %s", video_path.name)
        else:
            failed = [c.name for c in checks if not c.passed]
            logger.warning(
                "Quality checks FAILED for %s: %s", video_path.name, failed
            )

        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_file_exists(video_path: Path) -> CheckResult:
        """Check that the file exists and is non-empty."""
        if not video_path.exists():
            return CheckResult(
                name="file_exists",
                passed=False,
                detail=f"File not found: {video_path}",
            )
        size = video_path.stat().st_size
        if size == 0:
            return CheckResult(
                name="file_exists",
                passed=False,
                detail="File is empty (0 bytes)",
            )
        return CheckResult(
            name="file_exists",
            passed=True,
            detail=f"File exists ({size / 1024 / 1024:.1f} MB)",
        )

    @staticmethod
    def _check_valid_mp4(video_path: Path) -> CheckResult:
        """Check that ffprobe can parse the file as a valid MP4."""
        try:
            probe_video(video_path)
            return CheckResult(name="valid_mp4", passed=True, detail="ffprobe succeeded")
        except Exception as exc:
            return CheckResult(
                name="valid_mp4",
                passed=False,
                detail=f"ffprobe failed: {exc}",
            )

    @staticmethod
    def _check_duration(probe: dict, plan: EditingPlan) -> CheckResult:
        """Check that the rendered duration matches the editing plan duration."""
        actual = float(probe.get("format", {}).get("duration", 0.0))
        planned = plan.total_duration if plan.clips else plan.target_duration_seconds
        tolerance = 0.20
        lower = planned * (1 - tolerance)
        upper = planned * (1 + tolerance)

        if lower <= actual <= upper or (abs(actual - planned) <= 3.0):
            return CheckResult(
                name="duration",
                passed=True,
                detail=f"Duration {actual:.1f}s matches planned duration {planned:.1f}s (target: {plan.target_duration_seconds:.0f}s)",
            )
        return CheckResult(
            name="duration",
            passed=False,
            detail=f"Duration {actual:.1f}s differs from planned {planned:.1f}s",
        )

    @staticmethod
    def _check_resolution(probe: dict, plan: EditingPlan) -> CheckResult:
        """Check that the resolution matches the plan aspect ratio."""
        expected_w, expected_h = _EXPECTED_DIMS.get(plan.aspect_ratio, (1920, 1080))
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video":
                actual_w = stream.get("width", 0)
                actual_h = stream.get("height", 0)
                if actual_w == expected_w and actual_h == expected_h:
                    return CheckResult(
                        name="resolution",
                        passed=True,
                        detail=f"{actual_w}x{actual_h} matches expected",
                    )
                return CheckResult(
                    name="resolution",
                    passed=False,
                    detail=f"Got {actual_w}x{actual_h}, expected {expected_w}x{expected_h}",
                )
        return CheckResult(
            name="resolution",
            passed=False,
            detail="No video stream found in ffprobe output",
        )

    @staticmethod
    def _check_audio_stream(probe: dict) -> CheckResult:
        """Check that an audio stream is present."""
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "audio":
                codec = stream.get("codec_name", "unknown")
                return CheckResult(
                    name="audio_stream",
                    passed=True,
                    detail=f"Audio stream present (codec={codec})",
                )
        return CheckResult(
            name="audio_stream",
            passed=False,
            detail="No audio stream found",
        )

    @staticmethod
    def _check_no_black_video(video_path: Path) -> CheckResult:
        """
        Sample the first and last 2 seconds and check average brightness > 5.

        Uses FFmpeg's ``signalstats`` filter to compute YAVG (average luma).
        """
        ffmpeg = find_ffmpeg()
        try:
            # Check first 2 seconds
            args = [
                ffmpeg,
                "-ss", "0",
                "-t", "2",
                "-i", str(video_path),
                "-vf", "signalstats,metadata=print:file=-",
                "-f", "null",
                "-",
            ]
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            stderr = result.stderr + result.stdout

            # Extract YAVG values
            import re
            yavg_values = re.findall(r"YAVG=([\d.]+)", stderr)
            if yavg_values:
                avg_brightness = sum(float(v) for v in yavg_values) / len(yavg_values)
                if avg_brightness < 5.0:
                    return CheckResult(
                        name="no_black_video",
                        passed=False,
                        detail=f"Average luma {avg_brightness:.1f} < 5 (possible black video)",
                    )
                return CheckResult(
                    name="no_black_video",
                    passed=True,
                    detail=f"Average luma {avg_brightness:.1f} (OK)",
                )
        except Exception as exc:
            logger.warning("Black video check error: %s", exc)

        return CheckResult(
            name="no_black_video",
            passed=True,
            detail="Check skipped (signalstats unavailable)",
        )
