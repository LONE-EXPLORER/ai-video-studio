# AI Video Studio — System Architecture

## 1. Overview
AI Video Studio is a local-first, privacy-focused video editing application designed specifically for Apple Silicon macOS machines. It interprets high-level editing prompts, plans cuts using an LLM based on source timestamps, and executes automated rendering pipelines with FFmpeg and MLX Whisper.

## 2. Core Architectural Principles
- **Separation of Concerns:** Antigravity is strictly the development agent; the running studio application connects independently to local Ollama or optional Gemini API.
- **Local-First & Private:** Source footage never leaves the machine. No media is sent to cloud services by default.
- **Deterministic Editing:** LLMs never run raw shell commands. The LLM only returns structured, Pydantic-validated JSON plans (`EditingPlan`).
- **Sequential Resource Management:** Execution occurs in strict stages to conserve unified memory within a 24 GB hardware envelope.

## 3. Pipeline Stages
1. **Ingest & Validate:** Check video container, probe codecs, resolution, duration with FFprobe, copy into job directory without modifying original.
2. **Audio Extraction & Transcription:** Convert audio to 16kHz WAV, transcribe using MLX Whisper (Metal accelerated) with word-level timestamps.
3. **Visual Analysis:** Extract keyframes using FFmpeg, detect scene changes, assemble contact sheets.
4. **Editorial Intelligence:** LLM analyzes transcript + user prompt to select real timestamp ranges and assign zoom/pacing logic.
5. **Editing Execution:** Cut clips, assemble via FFmpeg concat demuxer, normalize audio.
6. **Caption Generation:** Build word-level highlighted ASS/SRT subtitles and burn into video.
7. **Rendering & Export:** Render H.264 MP4 with VideoToolbox hardware acceleration hint.
8. **Automated Quality Checks:** Validate duration, audio presence, absence of black frames.
9. **Web Preview:** Generate lightweight, faststart MP4 for instant browser playback.

## 4. Provider Abstraction
All AI intelligence runs through `BaseProvider`:
- `OllamaProvider`: Local Ollama REST API (`/api/generate`, `/api/tags`).
- `GeminiProvider`: Optional cloud fallback (disabled by default, requires explicit key).
