"""
Push-to-talk speech input, transcribed locally with faster-whisper.

Offline and English-only, matching the project's local-Ollama setup.
"""

import sys

SAMPLE_RATE = 16000
MODEL_SIZE = "base.en"


class SpeechTranscriber:
    """Records mic audio (push-to-talk) and transcribes it locally."""

    def __init__(self, console=None):
        self.console = console
        try:
            import sounddevice
            from faster_whisper import WhisperModel
        except ImportError as e:
            sys.exit(
                f"Voice mode needs extra packages ({e.name}).\n"
                "Install them with:  uv sync"
            )

        self._sd = sounddevice
        if console:
            console.print("[dim]Loading speech model (first run downloads it)…[/dim]")
        # int8 keeps it fast and light on CPU / Apple Silicon.
        self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    def listen(self) -> str:
        """Enter to start recording, Enter to stop. Returns transcribed text."""
        import numpy as np

        input("\n🎤 Press Enter to start recording…")
        frames = []

        def callback(indata, frame_count, time_info, status):
            frames.append(indata.copy())

        with self._sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
        ):
            input("🔴 Recording… press Enter to stop.")

        if not frames:
            return ""

        audio = np.concatenate(frames, axis=0).flatten()
        segments, _ = self.model.transcribe(audio, language="en")
        return " ".join(seg.text for seg in segments).strip()
