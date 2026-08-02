import re

# Vietnamese is written with each syllable as a whitespace-separated token,
# so speaking rate is measured in syllables per second (no phonemizer needed).
_WORD_RE = re.compile(r"\w", flags=re.UNICODE)


def _count_syllables(text):
    """Vietnamese syllables = whitespace-separated tokens that contain a word char."""
    if not text:
        return 0
    return sum(1 for tok in text.split() if _WORD_RE.search(tok))


def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):
    if isinstance(batch[text_column_name], list):
        speaking_rates = []
        if "speech_duration" in batch:
            for text, audio_duration in zip(batch[text_column_name], batch["speech_duration"]):
                audio_duration = audio_duration if audio_duration != 0 else 0.01
                speaking_rates.append(_count_syllables(text) / audio_duration)
        else:
            for text, audio in zip(batch[text_column_name], batch[audio_column_name]):
                audio_length = len(audio["array"].squeeze()) / audio["sampling_rate"]
                speaking_rates.append(_count_syllables(text) / audio_length)
        batch["speaking_rate"] = speaking_rates
    else:
        if "speech_duration" in batch:
            audio_length = batch["speech_duration"] if batch["speech_duration"] != 0 else 0.01
        else:
            audio = batch[audio_column_name]
            audio_length = len(audio["array"].squeeze()) / audio["sampling_rate"]
        batch["speaking_rate"] = _count_syllables(batch[text_column_name]) / audio_length

    return batch
