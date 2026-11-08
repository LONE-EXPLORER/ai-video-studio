"""
app/analysis/broll_pool.py — Local B-roll asset catalog and index manager.

Loads, validates, and queries curated B-roll clips from assets/broll/broll_index.json.
Provides tag-based and category-based retrieval for semantic video cutaways.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BROLL_DIR = PROJECT_ROOT / "assets" / "broll"
DEFAULT_INDEX_PATH = DEFAULT_BROLL_DIR / "broll_index.json"


@dataclass
class BrollAsset:
    """Metadata representing a single local B-roll clip."""

    id: str
    filename: str
    relative_path: str
    category: str
    tags: List[str]
    duration: float
    width: int = 1080
    height: int = 1920
    resolution: str = "1080x1920"
    aspect_ratio: str = "9:16"
    fps: int = 30
    color_space: str = "bt709"
    description: str = ""
    resolved_path: Optional[Path] = None

    @property
    def path(self) -> Path:
        """Absolute or resolved filesystem path to the asset file."""
        if self.resolved_path and self.resolved_path.exists():
            return self.resolved_path
        return PROJECT_ROOT / self.relative_path

    @property
    def file_exists(self) -> bool:
        """True if asset file exists on disk and is non-empty."""
        p = self.path
        return p.is_file() and p.stat().st_size > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "relative_path": self.relative_path,
            "category": self.category,
            "tags": list(self.tags),
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "aspect_ratio": self.aspect_ratio,
            "fps": self.fps,
            "color_space": self.color_space,
            "description": self.description,
        }


class BrollCatalog:
    """
    Catalog for managing and querying local B-roll assets.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        project_root: Optional[Path] = None,
        auto_provision: bool = False,
    ) -> None:
        self.project_root = project_root or PROJECT_ROOT
        self.index_path = index_path or DEFAULT_INDEX_PATH
        self._assets: Dict[str, BrollAsset] = {}
        self._loaded: bool = False

        if auto_provision and (not self.index_path.exists() or self.index_path.stat().st_size == 0):
            try:
                from scripts.provision_broll_pool import provision_broll_pool
                provision_broll_pool(self.index_path.parent)
            except Exception as exc:
                logger.warning("Could not auto-provision B-roll pool: %s", exc)

        self.reload()

    def reload(self) -> None:
        """Reload the catalog from index_path."""
        self._assets.clear()
        if not self.index_path.exists() or self.index_path.stat().st_size == 0:
            logger.info("B-roll index file '%s' does not exist or is empty.", self.index_path)
            self._loaded = True
            return

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse B-roll index JSON at '%s': %s", self.index_path, exc)
            raise

        assets_raw = data.get("assets", [])
        base_dir = self.index_path.parent

        for item in assets_raw:
            asset_id = str(item.get("id", "")).strip()
            if not asset_id:
                continue

            duration = float(item.get("duration", 0.0))
            if duration <= 0.0:
                logger.warning("Skipping B-roll asset '%s': duration <= 0 (%.2f)", asset_id, duration)
                continue

            filename = item.get("filename", f"{asset_id}.mp4")
            rel_path = item.get("relative_path") or f"assets/broll/{filename}"
            resolved = base_dir / filename if (base_dir / filename).exists() else (self.project_root / rel_path)

            raw_tags = item.get("tags", [])
            normalized_tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()]

            asset = BrollAsset(
                id=asset_id,
                filename=filename,
                relative_path=rel_path,
                category=str(item.get("category", "general")).strip(),
                tags=normalized_tags,
                duration=duration,
                width=int(item.get("width", 1080)),
                height=int(item.get("height", 1920)),
                resolution=str(item.get("resolution", "1080x1920")),
                aspect_ratio=str(item.get("aspect_ratio", "9:16")),
                fps=int(item.get("fps", 30)),
                color_space=str(item.get("color_space", "bt709")),
                description=str(item.get("description", "")),
                resolved_path=resolved,
            )
            self._assets[asset_id] = asset

        self._loaded = True
        logger.info("Loaded %d B-roll asset(s) from %s", len(self._assets), self.index_path)

    @property
    def total_count(self) -> int:
        return len(self._assets)

    def get_all(self, require_exists: bool = False) -> List[BrollAsset]:
        """Return all catalog assets, optionally filtering to only those on disk."""
        if require_exists:
            return [a for a in self._assets.values() if a.file_exists]
        return list(self._assets.values())

    def get_asset(self, asset_id: str) -> Optional[BrollAsset]:
        """Fetch asset by exact ID."""
        return self._assets.get(asset_id)

    def find_by_category(self, category: str, require_exists: bool = True) -> List[BrollAsset]:
        """Return assets matching category."""
        target = category.strip().lower()
        assets = [a for a in self._assets.values() if a.category.lower() == target]
        if require_exists:
            assets = [a for a in assets if a.file_exists]
        return assets

    def find_by_tags(self, query_tags: List[str], require_exists: bool = True) -> List[Tuple[BrollAsset, int]]:
        """
        Rank assets by count of overlapping tags.

        Returns
        -------
        List[Tuple[BrollAsset, int]]
            Ranked list of (asset, match_count) in descending order of overlap.
        """
        query_set = {t.strip().lower() for t in query_tags if t.strip()}
        if not query_set:
            return []

        results: List[Tuple[BrollAsset, int]] = []
        for asset in self.get_all(require_exists=require_exists):
            overlap = len(query_set.intersection(asset.tags))
            if overlap > 0:
                results.append((asset, overlap))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
