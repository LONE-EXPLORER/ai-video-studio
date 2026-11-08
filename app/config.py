"""
app/config.py — Application configuration, paths, settings.

Reads from environment variables and an optional .env file.
Exposes a singleton config instance via get_config().
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Ensure Homebrew and standard paths are in PATH for subprocesses (like mlx_whisper)
for p in ["/opt/homebrew/bin", "/usr/local/bin"]:
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{p}:{os.environ.get('PATH', '')}"

# Load .env from project root (walk up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_FILE, override=False)


@dataclass
class AppConfig:
    """Central application configuration loaded from environment variables."""

    # --- Ollama ---
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral-small:24b"))
    ollama_fallback_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_FALLBACK_MODEL", "qwen2.5-coder:14b")
    )

    # --- Gemini (optional) ---
    gemini_enabled: bool = field(
        default_factory=lambda: os.getenv("GEMINI_ENABLED", "false").lower() == "true"
    )
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))

    # --- App settings ---
    app_port: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "8501")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    max_video_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_VIDEO_SIZE_MB", "2000"))
    )

    # --- Directories ---
    jobs_dir_raw: str = field(default_factory=lambda: os.getenv("JOBS_DIR", "./jobs"))

    # --- Whisper ---
    whisper_model: str = field(
        default_factory=lambda: os.getenv(
            "WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"
        )
    )
    whisper_fallback_model: str = field(
        default_factory=lambda: os.getenv(
            "WHISPER_FALLBACK_MODEL", "mlx-community/whisper-small"
        )
    )

    @property
    def jobs_dir(self) -> Path:
        """Return the absolute path to the jobs directory."""
        raw = Path(self.jobs_dir_raw)
        if raw.is_absolute():
            return raw
        return (_PROJECT_ROOT / raw).resolve()

    @property
    def db_path(self) -> Path:
        """Return absolute path to the SQLite database file."""
        return self.jobs_dir / "studio.db"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return the singleton AppConfig, constructing it on first call."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def get_jobs_dir() -> Path:
    """Return the absolute jobs directory, creating it if it does not exist."""
    jobs_dir = get_config().jobs_dir
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir
