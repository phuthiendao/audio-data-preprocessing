"""Forced alignment (audio <-> transcript) with stable-ts + Whisper.

Independent of the profile pipeline (main.py process). Produces its own JSONL
of word/segment timestamps, joinable to data_final.jsonl by `filename`.

Run with the `stable-ts` conda env (has stable-ts / whisper), e.g.:

    python alignment/align.py \
      --input data --transcript data \
      --output output/data_final_alignment.jsonl \
      --model base --language vi
"""
import argparse
import glob
import json
import os

import soundfile as sf
import stable_whisper

AUDIO_EXTS = ("*.wav", "*.mp3", "*.m4a", "*.flac")

# Keep only samples whose audio duration is within these bounds (seconds).
MIN_DURATION = 1.0
MAX_DURATION = 30.0


def find_pairs(audio_dir, transcript_dir):
    """Pair each audio file with its <name>.txt transcript. Skips audio without a
    transcript and audio whose duration is outside [MIN_DURATION, MAX_DURATION]."""
    pairs, skipped = [], 0
    for ext in AUDIO_EXTS:
        for wav in glob.glob(os.path.join(audio_dir, "**", ext), recursive=True):
            base = os.path.splitext(os.path.basename(wav))[0]
            txt = os.path.join(transcript_dir, base + ".txt")
            if not os.path.exists(txt):
                continue  # no .txt pair
            try:
                info = sf.info(wav)
                dur = info.frames / info.samplerate
            except Exception:
                skipped += 1
                continue
            if not (MIN_DURATION <= dur <= MAX_DURATION):
                skipped += 1
                continue
            pairs.append((wav, txt))
    if skipped:
        print(f"[skip] {skipped} file(s): duration outside [{MIN_DURATION}s, {MAX_DURATION}s]")
    return sorted(pairs)


def load_done(output_path):
    """Filenames already aligned (for resume)."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["filename"])
                except Exception:
                    pass
    return done


def result_to_record(filename, result, language):
    words = [
        {
            "word": w.word,
            "start": round(w.start, 3),
            "end": round(w.end, 3),
        }
        for seg in result.segments
        for w in seg.words
    ]
    segments = [
        {"id": i, "start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text}
        for i, seg in enumerate(result.segments)
    ]
    return {
        "filename": filename,
        "language": language,
        "duration": round(segments[-1]["end"], 3) if segments else 0.0,
        "num_words": len(words),
        "words": words,
        "segments": segments,
    }


def main():
    ap = argparse.ArgumentParser(description="Forced alignment (stable-ts + Whisper)")
    ap.add_argument("--input", required=True, help="Audio directory")
    ap.add_argument("--transcript", required=True, help="Transcript directory (<name>.txt)")
    ap.add_argument("--output", required=True, help="Output JSONL")
    ap.add_argument("--model", default="base", help="Whisper model: base/small/medium/large-v3")
    ap.add_argument("--language", default="vi", help="Language code")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto-detect)")
    args = ap.parse_args()

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    pairs = find_pairs(args.input, args.transcript)
    done = load_done(args.output)
    todo = [(w, t) for (w, t) in pairs if os.path.basename(w) not in done]
    print(f"pairs={len(pairs)} done={len(done)} todo={len(todo)} | model={args.model} device={device}")

    if not todo:
        print("Nothing to do.")
        return

    print(f"Loading Whisper '{args.model}' on {device} ...")
    model = stable_whisper.load_model(args.model, device=device)

    ok = fail = 0
    with open(args.output, "a", encoding="utf-8") as f_out:
        for wav, txt in todo:
            fname = os.path.basename(wav)
            try:
                text = open(txt, "r", encoding="utf-8").read().strip()
                if not text:
                    print(f"[skip] {fname}: empty transcript")
                    fail += 1
                    continue
                result = model.align(wav, text, language=args.language)
                record = result_to_record(fname, result, args.language)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                ok += 1
                print(f"[ok] {fname}: {record['num_words']} words, {record['duration']}s ({ok + fail}/{len(todo)})")
            except Exception as e:
                fail += 1
                print(f"[fail] {fname}: {e}")

    print(f"Done. ok={ok} fail={fail} -> {args.output}")


if __name__ == "__main__":
    main()
