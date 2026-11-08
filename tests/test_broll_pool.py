"""
tests/test_broll_pool.py — Unit tests for B-roll Asset Pool and BrollCatalog.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.broll_pool import BrollAsset, BrollCatalog
from app.utils.ffmpeg_utils import probe_video


def test_catalog_loads_default_pool():
    """Verify catalog loads default local pool with all 6 categories."""
    catalog = BrollCatalog()
    assert catalog.total_count == 6

    expected_categories = {
        "warning_trap",
        "geopolitics_conflict",
        "technology_cyber",
        "finance_money",
        "law_justice",
        "abstract_motion",
    }
    loaded_categories = {a.category for a in catalog.get_all()}
    assert loaded_categories == expected_categories


def test_catalog_assets_exist_and_conform_to_specs():
    """Verify all 6 assets exist on disk, have duration 5.0s and 1080x1920 9:16 resolution."""
    catalog = BrollCatalog()
    assets = catalog.get_all(require_exists=True)
    assert len(assets) == 6

    for asset in assets:
        assert asset.file_exists, f"File missing on disk: {asset.path}"
        assert asset.width == 1080
        assert asset.height == 1920
        assert asset.aspect_ratio == "9:16"
        assert asset.duration == 5.0

        # Verify with ffprobe
        info = probe_video(asset.path)
        vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
        assert vstream["width"] == 1080
        assert vstream["height"] == 1920
        # Audio track must be absent or muted
        astreams = [s for s in info["streams"] if s["codec_type"] == "audio"]
        assert len(astreams) == 0, f"B-roll asset {asset.id} should not have audio stream"


def test_catalog_query_by_category():
    """Verify filtering by category works as expected."""
    catalog = BrollCatalog()
    tech_assets = catalog.find_by_category("technology_cyber")
    assert len(tech_assets) == 1
    assert tech_assets[0].id == "broll_tech_01"
    assert "apple silicon" in tech_assets[0].tags

    warning_assets = catalog.find_by_category("warning_trap")
    assert len(warning_assets) == 1
    assert warning_assets[0].id == "broll_warning_01"


def test_catalog_tag_ranking():
    """Verify multi-tag queries rank assets by number of matching tags."""
    catalog = BrollCatalog()
    # Query with tags matching geopolitics
    matches = catalog.find_by_tags(["missiles", "radar", "iran", "unknown_tag"])
    assert len(matches) > 0
    top_asset, score = matches[0]
    assert top_asset.id == "broll_geopolitics_01"
    assert score == 3  # missiles, radar, iran


def test_catalog_empty_index_handling(tmp_path: Path):
    """Verify empty index file produces empty catalog gracefully."""
    empty_index = tmp_path / "broll_index.json"
    empty_index.write_text(json.dumps({"assets": []}), encoding="utf-8")
    catalog = BrollCatalog(index_path=empty_index)
    assert catalog.total_count == 0
    assert catalog.get_all() == []
    assert catalog.get_asset("broll_tech_01") is None


def test_catalog_skips_zero_or_negative_duration(tmp_path: Path):
    """Verify assets with duration <= 0 are discarded."""
    index_file = tmp_path / "broll_index.json"
    index_data = {
        "assets": [
            {"id": "valid_01", "duration": 4.0, "filename": "v.mp4", "tags": ["a"]},
            {"id": "zero_dur", "duration": 0.0, "filename": "z.mp4", "tags": ["b"]},
            {"id": "neg_dur", "duration": -1.5, "filename": "n.mp4", "tags": ["c"]},
        ]
    }
    index_file.write_text(json.dumps(index_data), encoding="utf-8")
    catalog = BrollCatalog(index_path=index_file)
    assert catalog.total_count == 1
    assert catalog.get_asset("valid_01") is not None
    assert catalog.get_asset("zero_dur") is None
    assert catalog.get_asset("neg_dur") is None


def test_catalog_corrupted_json_raises(tmp_path: Path):
    """Verify corrupted index raises JSONDecodeError."""
    bad_index = tmp_path / "bad.json"
    bad_index.write_text("{ unquoted: bad ", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        BrollCatalog(index_path=bad_index)
