"""
scripts/download_whisper.py — Downloads and verifies MLX Whisper model weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from huggingface_hub import snapshot_download


def download_model(model_name: str = "mlx-community/whisper-large-v3-turbo") -> Path:
    print(f"\n📥 Verifying and downloading {model_name} (~1.6 GB)...")
    
    # Ensure HuggingFace lock directory exists
    repo_folder = f"models--{model_name.replace('/', '--')}"
    locks_dir = Path.home() / ".cache" / "huggingface" / "hub" / ".locks" / repo_folder
    locks_dir.mkdir(parents=True, exist_ok=True)

    try:
        path = snapshot_download(repo_id=model_name)
        print(f"✅ Model downloaded and cached at: {path}\n")
        return Path(path)
    except Exception as exc:
        print(f"❌ Failed to download Whisper model: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
