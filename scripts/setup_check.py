"""
scripts/setup_check.py — Verification script for AI Video Studio environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app.providers.ollama_provider import OllamaProvider
from app.utils.ffmpeg_utils import find_ffmpeg, find_ffprobe


def check_environment() -> bool:
    print("\n🔍 Running AI Video Studio Environment Verification...\n")
    all_ok = True

    # 1. Python version
    py_ver = sys.version.split()[0]
    py_ok = sys.version_info >= (3, 11)
    status_icon = "✅" if py_ok else "❌"
    print(f"{status_icon} Python Version: {py_ver} (required: >= 3.11)")
    if not py_ok:
        all_ok = False

    # 2. FFmpeg & FFprobe
    try:
        ffmpeg = find_ffmpeg()
        print(f"✅ FFmpeg Binary: {ffmpeg}")
    except Exception as e:
        print(f"❌ FFmpeg: {e}")
        all_ok = False

    try:
        ffprobe = find_ffprobe()
        print(f"✅ FFprobe Binary: {ffprobe}")
    except Exception as e:
        print(f"❌ FFprobe: {e}")
        all_ok = False

    # 3. Core Packages
    packages = [
        ("streamlit", "Streamlit"),
        ("pydantic", "Pydantic"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("requests", "Requests"),
    ]
    for mod_name, label in packages:
        try:
            __import__(mod_name)
            print(f"✅ {label}: Available")
        except ImportError:
            print(f"❌ {label}: Missing")
            all_ok = False

    # 4. Whisper Backend
    whisper_found = False
    try:
        import mlx_whisper
        print("✅ Whisper Backend: MLX Whisper (Metal accelerated)")
        whisper_found = True
    except ImportError:
        try:
            import faster_whisper
            print("🟡 Whisper Backend: faster-whisper (Fallback)")
            whisper_found = True
        except ImportError:
            print("❌ Whisper Backend: Neither mlx_whisper nor faster_whisper is available.")
            all_ok = False

    # 5. Ollama Status
    try:
        from app.config import get_config
        cfg = get_config()
        provider = OllamaProvider(host=cfg.ollama_host, model=cfg.ollama_model)
        if provider.is_available():
            models = provider.get_available_models()
            print(f"✅ Ollama Service: Online (Available models: {', '.join(models) if models else 'None'})")
        else:
            print(f"🟡 Ollama Service: Not responding at {provider._host} (Launch Ollama.app)")
    except Exception as e:
        print(f"🟡 Ollama Check: {e}")

    print("\n" + ("=" * 50))
    if all_ok:
        print("🎉 Environment verification PASSED. AI Video Studio is ready to run!")
    else:
        print("⚠️ Environment verification completed with missing dependencies.")
    print("=" * 50 + "\n")
    return all_ok


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
