"""
Test silence trimming on the video.
"""
from pathlib import Path
from app.editing.silence_remover import get_speech_segments, detect_silence

audio_path = Path("jobs/prod-ffa5bcba-a944-4c6e-815b-24f890fc941c/intermediate/audio.wav")
silences = detect_silence(audio_path, min_silence_duration=0.15, silence_threshold_db=-30.0)
print(f"Detected {len(silences)} pauses > 0.15s")
speech_segs = get_speech_segments(audio_path, duration=60.13, min_silence_duration=0.15, silence_threshold_db=-30.0)
print(f"Calculated {len(speech_segs)} active speech segments:")
total_speech = sum(end - start for start, end in speech_segs)
print(f"Total tight speech duration: {total_speech:.2f}s (out of 60.13s)")
for i, (s, e) in enumerate(speech_segs[:10]):
    print(f"  Segment {i+1}: {s:.2f}s -> {e:.2f}s (dur: {e-s:.2f}s)")
