"""
app/transcription/whisper_engine.py — MLX Whisper wrapper with faster-whisper fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from app.schemas.transcript import Transcript, TranscriptSegment, TranscriptWord
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TranscriptionEngine:
    """
    Speech-to-text engine that wraps either MLX Whisper (Apple Silicon) or
    faster-whisper (CPU/CUDA) as a fallback.

    Parameters
    ----------
    model_name:
        HuggingFace repo ID or local path for the Whisper model.
    use_mlx:
        If True (default), attempt to use mlx_whisper. Falls back to
        faster_whisper if mlx_whisper is not installed.
    """

    def __init__(self, model_name: str, use_mlx: bool = True) -> None:
        self._model_name = model_name
        self._use_mlx = use_mlx
        self._backend: str = "none"
        self._fw_model = None  # faster-whisper model, lazily loaded

        self._detect_backend()

    def _detect_backend(self) -> None:
        """Determine which backend to use and log the selection."""
        if self._use_mlx:
            try:
                import mlx_whisper  # noqa: F401

                self._backend = "mlx"
                logger.info("Transcription backend: MLX Whisper (model=%s)", self._model_name)
                return
            except ImportError:
                logger.warning(
                    "mlx_whisper not installed; falling back to faster-whisper."
                )

        try:
            from faster_whisper import WhisperModel  # noqa: F401

            self._backend = "faster_whisper"
            logger.info(
                "Transcription backend: faster-whisper (model=%s)", self._model_name
            )
        except ImportError:
            logger.error(
                "Neither mlx_whisper nor faster_whisper is installed. "
                "Install one to enable transcription."
            )
            self._backend = "none"

    def _load_fw_model(self):
        """Lazily load the faster-whisper model."""
        if self._fw_model is None:
            from faster_whisper import WhisperModel  # noqa: PLC0415

            logger.info("Loading faster-whisper model '%s' …", self._model_name)
            self._fw_model = WhisperModel(
                self._model_name,
                device="cpu",
                compute_type="int8",
            )
        return self._fw_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: Path) -> Transcript:
        """
        Transcribe the audio file at *audio_path* and return a :class:`Transcript`.

        Parameters
        ----------
        audio_path:
            Path to a 16 kHz mono WAV file (output of ffmpeg extract_audio).

        Returns
        -------
        Transcript

        Raises
        ------
        RuntimeError
            If no transcription backend is available.
        """
        if self._backend == "mlx":
            return self._transcribe_mlx(audio_path)
        elif self._backend == "faster_whisper":
            return self._transcribe_fw(audio_path)
        else:
            raise RuntimeError(
                "No transcription backend available. "
                "Install mlx-whisper or faster-whisper."
            )

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _transcribe_mlx(self, audio_path: Path) -> Transcript:
        """Transcribe using mlx_whisper with descriptive error reporting."""
        try:
            import mlx_whisper  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("mlx_whisper is not installed.") from exc

        logger.info("Transcribing with MLX Whisper: %s (model=%s)", audio_path, self._model_name)
        try:
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self._model_name,
                word_timestamps=True,
            )
            return self._parse_mlx_result(result)
        except Exception as exc:
            err_str = str(exc)
            if "IncompleteSnapshotError" in err_str or "ConnectError" in err_str or "weights" in err_str:
                raise RuntimeError(
                    f"MLX Whisper model '{self._model_name}' weights (~1.6 GB) are currently downloading or incomplete in cache. "
                    "Run 'python scripts/download_whisper.py' in your terminal to complete the download."
                ) from exc
            raise RuntimeError(f"MLX Whisper transcription failed: {exc}") from exc

    def _parse_mlx_result(self, result: dict) -> Transcript:
        """Convert mlx_whisper output dict to Transcript schema."""
        raw_segments = result.get("segments", [])
        segments: List[TranscriptSegment] = []

        for i, seg in enumerate(raw_segments):
            words: List[TranscriptWord] = []
            for w in seg.get("words", []):
                words.append(
                    TranscriptWord(
                        word=w.get("word", ""),
                        start=float(w.get("start", seg["start"])),
                        end=float(w.get("end", seg["end"])),
                        probability=float(w["probability"]) if "probability" in w else None,
                    )
                )
            segments.append(
                TranscriptSegment(
                    id=i,
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg.get("text", ""),
                    words=words,
                )
            )

        duration = float(result.get("duration", segments[-1].end if segments else 0.0))
        language = result.get("language", "en")
        return Transcript(language=language, duration=duration, segments=segments)

    def _transcribe_fw(self, audio_path: Path) -> Transcript:
        """Transcribe using faster-whisper."""
        model = self._load_fw_model()
        logger.info("Transcribing with faster-whisper: %s", audio_path)

        fw_segments_gen, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
        )

        segments: List[TranscriptSegment] = []
        for i, seg in enumerate(fw_segments_gen):
            words: List[TranscriptWord] = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        TranscriptWord(
                            word=w.word,
                            start=float(w.start),
                            end=float(w.end),
                            probability=float(w.probability) if w.probability is not None else None,
                        )
                    )
            segments.append(
                TranscriptSegment(
                    id=i,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text,
                    words=words,
                )
            )

        duration = float(info.duration) if hasattr(info, "duration") else (
            segments[-1].end if segments else 0.0
        )
        return Transcript(
            language=info.language,
            duration=duration,
            segments=segments,
        )
