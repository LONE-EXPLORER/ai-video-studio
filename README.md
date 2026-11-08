# 🎬 AI Video Studio

A complete, practical, local-first AI video editing studio built for Apple Silicon Macs.

Turn long-form source videos into polished social shorts, reels, or structured edits simply by typing natural language prompts.

---

## Features
- **Local-first AI:** Runs on your Mac using **Ollama** (`qwen2.5:7b` or `llama3.2`).
- **Apple Silicon Speech Recognition:** Native Metal acceleration via **MLX Whisper** with word-level timestamps.
- **Deterministic & Safe Video Pipeline:** FFmpeg subprocess pipeline validating real timestamps with Pydantic schemas. No raw shell execution.
- **Animated Captions:** Instagram/Reels-style word-by-word karaoke highlighting (ASS format).
- **Subtle Visual Effects:** Automatic zooms on key statements and jump-cut smoothing.
- **Automated Quality Checks:** Verifies export duration, audio streams, and absence of black frames.
- **Interactive Web Interface:** Streamlit UI for uploading, prompt tweaking, live progress, and one-click exports.

---

## Hardware & Environment Requirements
- **Machine:** Apple Silicon Mac (M-series chip, 24 GB unified memory recommended).
- **macOS:** 14+ (macOS 27 confirmed).
- **Dependencies:** Python 3.12+, Homebrew, FFmpeg, Ollama.

---

## Quickstart

### 1. Launch Ollama
Make sure Ollama is running:
```bash
open /Applications/Ollama.app
```
To pull the recommended editorial model (if not already downloaded):
```bash
ollama pull qwen2.5:7b
```

### 2. Activate Virtual Environment
```bash
cd /Users/pranjalshahi/.gemini/antigravity/scratch/ai-video-studio
source .venv/bin/activate
```

### 3. Verify Environment
```bash
python scripts/setup_check.py
```

### 4. Run Automated Test Suite
```bash
pytest tests/
```

### 5. Launch the Studio Application
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

---

## Directory Structure
```
ai-video-studio/
├── app/
│   ├── agents/          # Editorial LLM agent
│   ├── analysis/        # Keyframe extraction & scene detection
│   ├── captions/        # ASS/SRT caption generation & styling
│   ├── editing/         # FFmpeg clip cutting, zooming, audio processing
│   ├── pipeline/        # Stage runner & orchestrator
│   ├── providers/       # Ollama & Gemini provider abstraction
│   ├── quality/         # Automated export verification
│   ├── rendering/       # Final render & web preview generator
│   ├── schemas/         # Pydantic data models
│   ├── ui/              # Streamlit pages (Home, Processing, Preview, Export)
│   ├── utils/           # FFmpeg subprocess wrappers, SQLite db, safe file I/O
│   ├── config.py        # Central configuration
│   └── main.py          # Application entrypoint
├── scripts/
│   ├── generate_test_video.py  # Synthetic test video generator
│   └── setup_check.py          # Environment verification
├── tests/               # Pytest suite
└── docs/                # Architecture documentation
```
