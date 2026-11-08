#!/usr/bin/env python3
"""
scripts/provision_broll_pool.py — Procedural local B-roll asset generator & index provisioner.

Generates 6 vertical 1080x1920 @ 30fps CFR BT.709 clips across viral short-form retention
categories using Apple Silicon VideoToolbox (with automatic libx264 fallback).
Populates assets/broll/broll_index.json with semantic metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("provision_broll_pool")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BROLL_DIR = PROJECT_ROOT / "assets" / "broll"
DEFAULT_INDEX_PATH = DEFAULT_BROLL_DIR / "broll_index.json"

BROLL_SPECIFICATIONS: List[Dict[str, Any]] = [
    {
        "id": "broll_warning_01",
        "filename": "warning_trap_pulse.mp4",
        "category": "warning_trap",
        "tags": [
            "warning", "danger", "trap", "alert", "threat",
            "risk", "red button", "countdown", "gamble", "tactical"
        ],
        "duration": 5.0,
        "width": 1080,
        "height": 1920,
        "resolution": "1080x1920",
        "aspect_ratio": "9:16",
        "fps": 30,
        "color_space": "bt709",
        "description": "High-urgency red warning strobe with tactical grid overlay and hazard framing.",
        "lavfi_input": "color=c=black:s=1080x1920:r=30:d=5.0",
        "filter_complex": (
            "[0:v]geq=r='(160+70*sin(2*PI*T/1.2))*gt(mod(Y,12),4)':g='20':b='20',"
            "drawgrid=width=120:height=120:thickness=2:color=red@0.35,"
            "drawbox=x=120:y=800:w=840:h=320:color=red@0.8:t=6,setsar=1[vout]"
        ),
    },
    {
        "id": "broll_geopolitics_01",
        "filename": "geopolitics_radar_grid.mp4",
        "category": "geopolitics_conflict",
        "tags": [
            "military", "missiles", "radar", "war", "conflict",
            "geopolitics", "diplomacy", "iran", "president", "strike",
            "attack", "power plants"
        ],
        "duration": 5.0,
        "width": 1080,
        "height": 1920,
        "resolution": "1080x1920",
        "aspect_ratio": "9:16",
        "fps": 30,
        "color_space": "bt709",
        "description": "Concentric tactical radar sweep with latitude telemetry grid and green target blips.",
        "lavfi_input": "color=c=#04140b:s=1080x1920:r=30:d=5.0",
        "filter_complex": (
            "[0:v]drawgrid=width=90:height=90:thickness=2:color=#00ff66@0.2,"
            "drawbox=x=140:y=560:w=800:h=800:color=#00ff66@0.5:t=4,"
            "geq=r='10':g='30+120*pow(max(0,1-hypot(X-540,Y-960)/420),2)':b='20',setsar=1[vout]"
        ),
    },
    {
        "id": "broll_tech_01",
        "filename": "tech_code_matrix.mp4",
        "category": "technology_cyber",
        "tags": [
            "coding", "python", "developer", "apple silicon", "software",
            "technology", "ai", "matrix", "hardware", "algorithm"
        ],
        "duration": 5.0,
        "width": 1080,
        "height": 1920,
        "resolution": "1080x1920",
        "aspect_ratio": "9:16",
        "fps": 30,
        "color_space": "bt709",
        "description": "Dynamic matrix code stream with glowing cyan and emerald terminal characters.",
        "lavfi_input": "color=c=#030a12:s=1080x1920:r=30:d=5.0",
        "filter_complex": (
            "[0:v]geq=r='15':g='40+80*sin((X+Y*2+T*300)/100)':b='120+60*cos((X-Y+T*200)/150)',"
            "drawgrid=width=40:height=40:thickness=1:color=#00ccff@0.2,setsar=1[vout]"
        ),
    },
    {
        "id": "broll_money_01",
        "filename": "finance_stock_chart.mp4",
        "category": "finance_money",
        "tags": [
            "finance", "money", "growth", "business", "stock",
            "crypto", "wealth", "economy", "profit", "market",
            "investment", "dollar"
        ],
        "duration": 5.0,
        "width": 1080,
        "height": 1920,
        "resolution": "1080x1920",
        "aspect_ratio": "9:16",
        "fps": 30,
        "color_space": "bt709",
        "description": "Ascending financial candlestick chart with neon green trendline and grid telemetry.",
        "lavfi_input": "color=c=#080d16:s=1080x1920:r=30:d=5.0",
        "filter_complex": (
            "[0:v]drawgrid=width=100:height=100:thickness=1:color=#ffffff@0.15,"
            "drawbox=x=200:y=1100:w=120:h=300:color=#00ff88@0.7:t=fill,"
            "drawbox=x=380:y=900:w=120:h=500:color=#00ff88@0.7:t=fill,"
            "drawbox=x=560:y=1000:w=120:h=400:color=#ff3344@0.7:t=fill,"
            "drawbox=x=740:y=700:w=120:h=700:color=#00ff88@0.7:t=fill,setsar=1[vout]"
        ),
    },
    {
        "id": "broll_law_01",
        "filename": "law_justice_scales.mp4",
        "category": "law_justice",
        "tags": [
            "law", "court", "legal", "justice", "judge",
            "contract", "promise", "verdict", "trial", "ceasefire", "agreement"
        ],
        "duration": 5.0,
        "width": 1080,
        "height": 1920,
        "resolution": "1080x1920",
        "aspect_ratio": "9:16",
        "fps": 30,
        "color_space": "bt709",
        "description": "Gold scales of justice and legal pillar silhouette against dark navy vignette.",
        "lavfi_input": "color=c=#0f1422:s=1080x1920:r=30:d=5.0",
        "filter_complex": (
            "[0:v]drawbox=x=520:y=600:w=40:h=700:color=#d4af37@0.85:t=fill,"
            "drawbox=x=280:y=680:w=520:h=24:color=#d4af37@0.85:t=fill,"
            "drawbox=x=240:y=800:w=160:h=20:color=#d4af37@0.75:t=fill,"
            "drawbox=x=680:y=760:w=160:h=20:color=#d4af37@0.75:t=fill,setsar=1[vout]"
        ),
    },
    {
        "id": "broll_abstract_01",
        "filename": "abstract_kinetic_tunnel.mp4",
        "category": "abstract_motion",
        "tags": [
            "abstract", "energy", "speed", "motion", "focus",
            "pattern interrupt", "hypnotic", "tunnel", "flow", "burst"
        ],
        "duration": 5.0,
        "width": 1080,
        "height": 1920,
        "resolution": "1080x1920",
        "aspect_ratio": "9:16",
        "fps": 30,
        "color_space": "bt709",
        "description": "High-velocity chromatic tunnel vortex with pulsing depth perspective.",
        "lavfi_input": "mandelbrot=size=1080x1920:rate=30:maxiter=80",
        "filter_complex": "[0:v]setsar=1[vout]",
    },
]


def find_ffmpeg_binary() -> str:
    """Find local ffmpeg binary."""
    # Check well-known Apple Silicon Homebrew path first
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"]:
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError("FFmpeg executable could not be found.")


def has_videotoolbox(ffmpeg_bin: str) -> bool:
    """Check if h264_videotoolbox encoder is available."""
    try:
        res = subprocess.run(
            [ffmpeg_bin, "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return "h264_videotoolbox" in res.stdout
    except Exception:
        return False


def render_clip(
    spec: Dict[str, Any],
    output_path: Path,
    ffmpeg_bin: str,
    use_videotoolbox: bool = True,
) -> None:
    """Render a single procedural clip."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder_args = (
        ["-c:v", "h264_videotoolbox", "-b:v", "4500k", "-pix_fmt", "nv12"]
        if use_videotoolbox
        else ["-c:v", "libx264", "-crf", "22", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
    )

    is_mandelbrot = "mandelbrot" in spec.get("lavfi_input", "")
    input_args = (
        ["-f", "lavfi", "-i", spec["lavfi_input"], "-t", str(spec["duration"])]
        if is_mandelbrot
        else ["-f", "lavfi", "-i", spec["lavfi_input"]]
    )

    cmd = [
        ffmpeg_bin,
        "-y",
        *input_args,
        "-filter_complex", spec["filter_complex"],
        "-map", "[vout]",
        "-an",
        *encoder_args,
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as exc:
        if use_videotoolbox:
            logger.warning(
                "VideoToolbox encoding failed for '%s', falling back to libx264: %s",
                spec["filename"],
                exc.stderr.decode("utf-8", errors="replace")[-300:],
            )
            render_clip(spec, output_path, ffmpeg_bin, use_videotoolbox=False)
        else:
            raise RuntimeError(
                f"Failed to generate B-roll clip {spec['filename']}: "
                f"{exc.stderr.decode('utf-8', errors='replace')}"
            ) from exc


def provision_broll_pool(
    broll_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Ensure all 6 procedural B-roll clips and broll_index.json exist.

    Returns
    -------
    Path
        Path to broll_index.json.
    """
    target_dir = broll_dir or DEFAULT_BROLL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    index_path = target_dir / "broll_index.json"

    ffmpeg_bin = find_ffmpeg_binary()
    vt_available = has_videotoolbox(ffmpeg_bin)
    logger.info("FFmpeg: %s (VideoToolbox: %s)", ffmpeg_bin, vt_available)

    assets_metadata: List[Dict[str, Any]] = []

    for spec in BROLL_SPECIFICATIONS:
        clip_path = target_dir / spec["filename"]
        rel_path = f"assets/broll/{spec['filename']}"

        # Clean normalized lowercase tags
        clean_tags = [t.strip().lower() for t in spec["tags"] if t.strip()]

        asset_entry = {
            "id": spec["id"],
            "filename": spec["filename"],
            "relative_path": rel_path,
            "category": spec["category"],
            "tags": clean_tags,
            "duration": float(spec["duration"]),
            "width": int(spec["width"]),
            "height": int(spec["height"]),
            "resolution": spec["resolution"],
            "aspect_ratio": spec["aspect_ratio"],
            "fps": int(spec["fps"]),
            "color_space": spec["color_space"],
            "description": spec["description"],
        }
        assets_metadata.append(asset_entry)

        if force or not clip_path.exists() or clip_path.stat().st_size == 0:
            logger.info("Generating B-roll clip [%s] -> %s", spec["id"], clip_path.name)
            render_clip(spec, clip_path, ffmpeg_bin, use_videotoolbox=vt_available)
        else:
            logger.debug("Clip %s already exists, skipping generation", clip_path.name)

    index_payload = {
        "version": "1.0",
        "description": "AI Video Studio Local B-Roll Asset Pool Index",
        "base_directory": "assets/broll",
        "assets": assets_metadata,
    }

    if force or not index_path.exists() or index_path.stat().st_size == 0:
        index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
        logger.info("Wrote B-roll index to %s (%d assets)", index_path, len(assets_metadata))

    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision local B-roll asset pool.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BROLL_DIR,
        help="Directory to output B-roll assets and index",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all clips and index",
    )
    args = parser.parse_args()
    provision_broll_pool(broll_dir=args.output_dir, force=args.force)
    logger.info("B-roll pool provisioned successfully.")


if __name__ == "__main__":
    main()
