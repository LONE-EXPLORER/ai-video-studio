# Functional Document (FD) — AI Video Studio

## 1. Primary Intent & Vision

### Primary Objective
To build a **practical, production-ready, local-first AI-powered video editing studio** tailored specifically for an **Apple Silicon Mac Mini (24 GB Unified Memory, macOS)**.

The application allows a creator to:
1. Drop raw video footage into a modern graphical interface.
2. Type natural-language editing instructions (e.g., *"Turn this 5-minute video into a high-energy 45-second Instagram Reel, cut unnecessary pauses, add dynamic captions, and zoom on key points"*).
3. Let local AI models analyze the speech transcript, identify the strongest hook, plan cuts referencing **real source timestamps**, and execute professional video editing through FFmpeg.
4. Preview, review the AI editing plan, and export finished high-definition video assets—**with zero cloud dependencies and zero recurring subscription fees**.

---

## 2. Core Non-Negotiable Principles

- **Zero Cloud Dependence by Default:** Media processing, speech-to-text, and editorial intelligence run entirely on the local Mac hardware.
- **Provider Independence:** Antigravity is strictly the development harness; the studio application functions independently with local Ollama (`qwen2.5:7b` / `llama3.2`) and an optional, opt-in Google AI Studio Gemini API fallback.
- **Deterministic & Safe Video Pipeline:** AI models NEVER execute raw shell commands. The LLM only produces structured, Pydantic-validated JSON plans (`EditingPlan`). All rendering is performed via safe subprocess argument arrays.
- **Original Media Preservation:** Original source videos are treated as read-only assets and are never modified or deleted.
- **Sequential Resource Management:** Execution occurs in strict sequential stages to conserve unified memory within the 24 GB hardware envelope.

---

## 3. System Architecture & Components

```
ai-video-studio/
├── app/
│   ├── agents/          # Editorial LLM agent (prompting, validation, retry loops)
│   ├── analysis/        # Frame extraction, scene detection, visual contact sheets
│   ├── captions/        # ASS/SRT caption generation with word-level highlighting
│   ├── editing/         # FFmpeg clip cutting, silence removal, audio processing
│   ├── pipeline/        # Stage runner and end-to-end orchestrator
│   ├── providers/       # Ollama & Gemini provider abstraction
│   ├── quality/         # Automated export verification
│   ├── rendering/       # Final render and faststart web preview generator
│   ├── schemas/         # Pydantic models (EditingPlan, Transcript, VideoMetadata)
│   ├── ui/              # Streamlit graphical interface (Home, Processing, Preview, Export)
│   ├── utils/           # Subprocess wrappers, SQLite db, safe file I/O, logger
│   ├── config.py        # Central configuration
│   └── main.py          # Application entrypoint
├── launch.sh            # 1-click startup launcher
├── FD.md                # This Functional Document
├── README.md            # Quickstart and setup instructions
└── tests/               # Automated test suite
```

---

## 4. Phase-by-Phase Roadmap & Status Tracker

| Phase | Description | Status | Key Deliverables |
|---|---|---|---|
| **Phase 0** | Environment Audit | ✅ Complete | Hardware audit, Python 3.12, FFmpeg 9.0.2, Ollama inspection |
| **Phase 1** | Minimum Working Application | ✅ Complete | Full Streamlit UI, MLX Whisper, safe FFmpeg editing, 15/15 unit tests |
| **Phase 2** | Intelligent Editing | ✅ Complete | Silence detection (`silencedetect`), pause trimming, timestamp validation |
| **Phase 3** | Professional Presentation | ✅ Complete | Animated captions (karaoke tags), dynamic zoompan, audio ducking with BGM |
| **Phase 4** | Creative Additions | ✅ Complete | Background music auto-ducking, contact sheet scene analysis, DaVinci FCPXML export |
| **Phase 5** | Reliability & Finishing | ✅ Complete | Full end-to-end production verification on user video, live logging, intermediate caching |

---

## 5. Ongoing Log of Important Implementation Decisions

- **2026-11-05 (Production Verification with Real Footage):** Executed complete end-to-end production pipeline on user video `_Trump’s 5-Day Trap` (`media_1793829730655_4e653906.mp4`):
  - Ingest: 0.0s
  - MLX Whisper GPU transcription: 4.2s (25 segments with word-level timestamps)
  - Keyframe visual extraction: 0.4s (12 keyframes + contact sheet)
  - Mistral 24B editorial cut planning (`num_ctx: 8192`): 10 clips planned, hook identified
  - Cut execution & dynamic punch-in zooms: 115% hook, alternating 108% scale
  - VideoToolbox hardware render: 9:16 1080x1920 MP4 (10.15 MB) with ducked ambient music
  - DaVinci Resolve FCPXML v1.9 timeline: generated with transform scale track
  - Web-optimized preview: generated and verified
- **2026-11-05 (DaVinci Resolve FCPXML Timeline Export):** Added `app/export/fcpxml_generator.py` producing standard FCPXML 1.9 files containing cut sequences, source media references, and scale transforms for color grading in DaVinci Resolve and Final Cut Pro.
- **2026-11-05 (Model Architecture & Context Window):** Resolved Ollama timeout by enforcing `num_ctx: 8192` and `num_predict: 1024` on `mistral-small:24b`, avoiding GBNF grammar truncation loops. Added automatic failover to `qwen2.5-coder:14b`.
- **2026-11-05 (Live Step Logging & Resilient Caching):** Added real-time timestamped event logging into `pipeline.log` visible live in the UI expander. Implemented intermediate caching so retrying any stage skips repeating completed Whisper or keyframe tasks.
- **2026-11-05 (Copyright Policy):** Media and music assets are strictly restricted to local royalty-free/public domain sources in `assets/music/` to prevent copyright strikes on social media.
- **2026-11-05 (Phase 1):** Built complete modular application structure with Pydantic schemas. Integrated Streamlit multi-page interface. Created synthetic test video pipeline. Verified 13 unit tests.
- **2026-11-05 (Phase 1 Fix):** Solved `PYTHONPATH` module discovery and port 8501 lingering process cleanup via `launch.sh`.
- **2026-11-05 (Phase 2 & 3):** 
  - Integrated `silence_remover.py` (FFmpeg `silencedetect` filter) for detecting speech pauses and silent gaps.
  - Implemented dynamic zoom effect via FFmpeg `zoompan` filter (1.00x → 1.05x linear ramp on highlighted statements).
  - Implemented background music ducking (`audio_processor.duck_music`) with automatic track detection from `assets/music/` and `-stream_loop -1` looping.
  - Added graceful fallback in caption burning to guarantee export delivery even on minimal FFmpeg builds while preserving full `.ass` and `.srt` downloads.
  - Expanded test suite to 15 passing tests.
