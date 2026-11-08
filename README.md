# 🎬 AI Video Studio

A complete, high-retention, local-first AI video editing studio built for Apple Silicon Macs with **Google AI Studio (Gemini)** hybrid intelligence.

Turn raw source videos into viral, broadcast-grade YouTube Shorts, TikToks, and Instagram Reels with **kinetic typography, dynamic punch-in zooms, animated geopolitical maps, financial charts, sound effects, and DaVinci Resolve multi-track export**.

---

## ⚡ The Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────┐
│             🧠 THE DIRECTOR: Google AI Studio               │
│        (Gemini 2.0 Flash / 1.5 Pro - 1M+ Context)           │
│  • Analyzes raw video & word-level speech transcript         │
│  • Plans hook concept (115% zoom), pacing, and sound cues   │
│  • Directs when & where to insert Maps, Charts, and B-Roll  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Structured JSON Plan
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          ⚙️ THE ENGINE: Local Apple Silicon Studio          │
│             (Mac Mini / Studio / MacBook Pro)               │
│  • Local Speech: MLX Whisper Large-v3-turbo (GPU Metal)     │
│  • Local Maps: Animated tactical map generator (Pillow)     │
│  • Local Charts: Progressive neon financial chart generator │
│  • Local Audio: SFX adelay mixer + sidechain compressor     │
│  • Hardware Render: Apple VideoToolbox (H.264 / HEVC)       │
│  • NLE Export: DaVinci Resolve 6-track FCPXML v1.9 timeline │
└─────────────────────────────────────────────────────────────┘
```

---

## Features
- **Google AI Studio Director:** Dedicated prompt & schema for Gemini to direct second-by-second maps, charts, footage, and audio ([See Director Prompt Guide](docs/GOOGLE_AI_STUDIO_DIRECTOR_PROMPT.md)).
- **Local Map & Chart Generation:** Run `scripts/generate_animated_map.py` and `scripts/generate_animated_chart.py` to create 1080x1920 broadcast graphics locally in seconds.
- **Relentless Silence Trimming:** Automatically eliminates dead pauses (>0.15s) with 10ms micro-crossfades to prevent audio pops.
- **Cinematic Motion:** 115% hook punch-in on opening statement; alternating 108%/100% zooms on jump cuts with `setsar=1` stabilization.
- **Kinetic Typography:** Spoken-word highlight in Bright Yellow (`#FFCC00`) with rounded dark pill background and harmonic spring bounce (1.20× → 1.00×).
- **Dynamic Sound Effects & Ducking:** 25 auto-placed SFX (`impact`, `whoosh`, `riser`, `pop`) with `sidechaincompress` broadcast music ducking.
- **DaVinci Resolve Multi-Track XML:** Generates full `.fcpxml` with V1 A-roll, V2 B-roll, V3 captions, A1 dialogue, A2 SFX, A3 BGM.
- **Apple Silicon Hardware Acceleration:** 100% native Metal & VideoToolbox encoding.
- **Interactive Web UI:** Streamlit interface running locally at `http://localhost:8501`.

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
