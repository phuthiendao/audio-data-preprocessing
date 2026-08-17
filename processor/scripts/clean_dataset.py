"""Stage 0: pre-extraction filtering + transcript cleaning.

Reads an audio dir + transcript dir and writes a CLEAN dataset dir containing
only samples that pass every filter, each with a normalized .txt. Non-destructive:
the originals are never modified. Run BEFORE `process` (and point `alignment` at
the same clean dir for consistency).

Filters (checked cheap -> expensive so silent/broken files bail before VAD load):
  1. missing .txt pair            5. empty transcript after cleaning
  2. zero-byte .wav or .txt       6. VAD: < MIN_SPEECH_DURATION sec of speech
  3. duration outside [MIN, MAX]  (kept files get cleaned .txt + copied audio)
"""
import os
import shutil
import argparse
from collections import Counter

from processor.config import (
    MIN_AUDIO_DURATION, MAX_AUDIO_DURATION, MIN_SPEECH_DURATION,
)
from processor.utils.audio_io import get_audio_files, audio_duration_sec, load_audio_16k
from processor.utils.text_clean import clean_transcript, is_empty_transcript


def _txt_for(filepath, transcript_dir):
    base = os.path.splitext(os.path.basename(filepath))[0]
    return os.path.join(transcript_dir, base + ".txt")


def clean_dataset(audio_dir, transcript_dir, output_dir, keep_brackets=False):
    os.makedirs(output_dir, exist_ok=True)
    kept, skipped = 0, Counter()

    for filepath in get_audio_files(audio_dir):
        fname = os.path.basename(filepath)
        txt_path = _txt_for(filepath, transcript_dir)

        # 1. pair
        if not os.path.exists(txt_path):
            skipped["no .txt pair"] += 1
            continue
        # 2. zero-byte
        try:
            if os.path.getsize(filepath) == 0 or os.path.getsize(txt_path) == 0:
                skipped["zero-byte file"] += 1
                continue
        except OSError:
            skipped["unreadable file"] += 1
            continue
        # 3. duration
        try:
            dur = audio_duration_sec(filepath)
        except Exception:
            skipped["unreadable audio"] += 1
            continue
        if not (MIN_AUDIO_DURATION <= dur <= MAX_AUDIO_DURATION):
            skipped["duration out of range"] += 1
            continue
        # 4 + 5. clean text, drop if empty after cleaning
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                cleaned = clean_transcript(f.read(), strip_brackets=not keep_brackets)
        except Exception:
            skipped["unreadable transcript"] += 1
            continue
        if is_empty_transcript(cleaned):
            skipped["empty after cleaning"] += 1
            continue
        # 6. VAD (expensive: decodes audio)
        try:
            from processor.utils.vad import speech_duration_sec
            audio = load_audio_16k(filepath)
            speech = speech_duration_sec(audio)
        except ImportError:
            raise  # missing silero-vad: surface the install hint, don't swallow
        except Exception:
            skipped["vad error"] += 1
            continue
        if speech < MIN_SPEECH_DURATION:
            skipped["no/short speech (VAD)"] += 1
            continue

        # passed: write cleaned .txt + copy audio into the clean dir
        base = os.path.splitext(fname)[0]
        with open(os.path.join(output_dir, base + ".txt"), "w", encoding="utf-8") as f:
            f.write(cleaned + "\n")
        shutil.copy2(filepath, os.path.join(output_dir, fname))
        kept += 1
        print(f"[keep] {fname} (speech {speech:.2f}s)")

    total_skipped = sum(skipped.values())
    print(f"\nClean done -> {output_dir}")
    print(f"  kept:    {kept}")
    print(f"  skipped: {total_skipped}")
    for reason, n in skipped.most_common():
        print(f"    - {reason}: {n}")
    return kept, dict(skipped)


def main():
    ap = argparse.ArgumentParser(
        description="Stage 0: clean transcripts + pre-extraction filtering (VAD, duration, pairs)"
    )
    ap.add_argument("--input", required=True, help="Input audio directory")
    ap.add_argument("--transcript", default=None, help="Transcript dir (.txt); defaults to --input")
    ap.add_argument("--output", required=True, help="Output CLEAN dataset dir (cleaned .txt + kept audio)")
    ap.add_argument("--keep-brackets", action="store_true", help="Preserve [...] and (...) tags in transcript (e.g. for rich-text TTS models like CosyVoice)")
    args = ap.parse_args()
    clean_dataset(args.input, args.transcript or args.input, args.output, keep_brackets=args.keep_brackets)


if __name__ == "__main__":
    main()
