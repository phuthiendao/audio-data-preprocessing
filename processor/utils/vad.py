"""Silero-VAD speech detection for Stage 0 (pre-extraction).

Lazy singleton model load: importing this module is cheap and the rest of the
pipeline never needs silero-vad unless `clean` actually runs.
"""
from processor.config import SAMPLING_RATE

_model = None
_get_speech_timestamps = None


def _ensure_model():
    global _model, _get_speech_timestamps
    if _model is None:
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
        except ImportError as e:
            raise ImportError(
                "silero-vad is required for `clean` (VAD). Install it: pip install silero-vad"
            ) from e
        _model = load_silero_vad()
        _get_speech_timestamps = get_speech_timestamps
    return _model, _get_speech_timestamps


def speech_duration_sec(audio, sampling_rate=SAMPLING_RATE):
    """Total detected speech, in seconds, for a mono float32 waveform (numpy or tensor)."""
    import torch

    model, get_ts = _ensure_model()
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio, dtype=torch.float32)
    segments = get_ts(audio, model, sampling_rate=sampling_rate)
    return sum(s["end"] - s["start"] for s in segments) / float(sampling_rate)
