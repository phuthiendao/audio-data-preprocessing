# Preprocessing-Data-TTS

## Introduction

A preprocessing pipeline that turns speech **audio + transcript** into rich,
controllable-TTS-style annotations (Parler-TTS / Data-Speech style), tuned for
Vietnamese. Given an audio file and its transcript, the pipeline:

1. **Extracts a voice profile** — numeric acoustic/speaker attributes.
2. **Maps the profile to text labels** — human-readable descriptors.
3. **Generates a natural-language description** of the voice (via Gemini).

Under the hood it combines:

- **OpenSMILE** (eGeMAPSv02) — voice-quality, formant, spectral and loudness features.
- **penn / FCNF0++** — fundamental frequency (pitch) and its variation in semitones.
- **brouhaha** — signal-to-noise ratio (SNR) and reverberation (C50).
- **Whisper-based classifiers** — gender and regional accent.
- **Gemini** — final natural-language description.

Normalization is **two-layered**: feature → score uses fixed, physiologically
grounded ranges (dataset-independent), while score → text label uses
**data-driven quantile bins** computed from the dataset at convert time.

A separate **forced-alignment** tool (stable-ts + Whisper) produces word-level
timestamps aligned to the transcript (see `alignment/`).

## Installation

Create the environment (Python 3.10):

```bash
conda create -n preprocessing-data python=3.10
conda activate preprocessing-data
```

Install dependencies for your hardware — pick **one**:

```bash
# CPU
pip install -r requirements_cpu.txt

# GPU (CUDA 12.6)
pip install -r requirements_gpu.txt
```

Then install the remaining pieces (both CPU and GPU):

```bash
pip install --no-deps "brouhaha @ https://github.com/marianne-m/brouhaha-vad/archive/main.zip"
pip install -q -U google-genai
```

For step 3, set the Gemini API key:

```bash
export GOOGLE_API_KEY="your_api_key"
```

## Usage

### 0. Clean & Filter Dataset (Stage 0 — before extraction)

Normalizes transcripts and drops unusable samples into a **new, self-contained
clean directory** (cleaned `.txt` + kept audio) — originals are never modified.
Point steps 1 and 4 at this clean dir for consistency.

Filters: missing `.txt`/`.wav` pairs, zero-byte files, duration outside
`[1.0s, 30.0s]`, empty transcripts after cleaning, and silent / no-speech audio
(`< 1.0s` of speech via silero-VAD).

Cleaning: strips HTML and emojis; normalizes quotes and Vietnamese tone placement
(`hoà`→`hòa`, `thuỷ`→`thủy`); collapses repeated punctuation (`???`→`?`). By default
strips `[...]`/`(...)` tags, but passing `--keep-brackets` preserves them and
standardizes parenthesized tags `(...)` to `[...]` (e.g. `(cười)`→`[cười]`) for
rich-text / controllable TTS models (CosyVoice, ChatTTS).

```bash
python processor/main.py clean \
    --input <audio_dir> \
    --transcript <transcript_dir> \
    --output <clean_dir> \
    [--keep-brackets]
```

`--transcript` defaults to `--input` (when audio and `.txt` live in one folder).

### 1. Extract Voice Profiles

Audio → numeric voice profile (`.jsonl`). This is the only GPU-accelerated step.

```bash
# CPU (one file at a time — fastest & most stable on few cores)
python processor/main.py process \
    --input <audio_dir> \
    --transcript <transcript_dir> \
    --output <profiles.jsonl> \
    --sequential

# GPU (pipelined: parallel CPU workers + batched GPU inference)
python processor/main.py process \
    --input <audio_dir> \
    --transcript <transcript_dir> \
    --output <profiles.jsonl> \
    --workers 8 \
    --batch_size 32
```

> **Stage 2 (post-extraction filter).** Samples with a low signal-to-noise ratio
> (`SNR < 5.0 dB`) are automatically dropped from the output during this step.
> Thresholds live in `processor/config.py` (`SNR_MIN_DB`, `MIN_SPEECH_DURATION`).

### 2. Convert to Text Labels

Numeric profile → text labels via data-driven quantile bins. CPU-only (no GPU
difference).

```bash
python processor/main.py convert \
    --input <profiles.jsonl> \
    --output <profiles_text.jsonl>
```

### 3. Generate Descriptions

Text labels → natural-language description via Gemini (cloud API — no GPU
difference).

```bash
python prompts/generate_descriptions.py \
    --input <profiles_text.jsonl> \
    --output <descriptions.jsonl> \
    --audio-dir <audio_dir>
```

> **Tip — augment the description data.** The wording is driven entirely by the
> prompt in `prompts/gemini_prompt_template.py`. Edit it — or run several passes
> with different variants — to produce *diverse* captions for the same audio,
> which is useful as data augmentation for controllable TTS. A simple recipe is
> to drop one attribute group per variant:
> - **identity** — omit `gender` / `age_range` / `region_accent`
> - **voice** — omit the timbre / pitch / breathiness / resonance wording
> - **environment** — omit `recording_quality` / `reverberation`
>
> Each variant yields a differently-focused description for the same sample.

### 4. Forced Alignment (optional)

Align each transcript to its audio and export **word-level timestamps** to a
separate `.jsonl` (joinable to the profiles by `filename`). Independent of
steps 1–3 and runs in its own environment:

```bash
conda create -n stable-ts python=3.10 && conda activate stable-ts
pip install stable-ts openai-whisper
```

```bash
# CPU (small model — fast)
python alignment/align.py \
    --input <audio_dir> \
    --transcript <transcript_dir> \
    --output <alignment.jsonl> \
    --model base --language vi

# GPU (larger model — more accurate)
python alignment/align.py \
    --input <audio_dir> \
    --transcript <transcript_dir> \
    --output <alignment.jsonl> \
    --model large-v3 --language vi
```

The script auto-detects CPU/GPU; the practical CPU vs GPU choice is the model
size (`base` for CPU speed, `large-v3` for GPU accuracy).

## Project Structure

```
preprocessing-data-tts/
├── processor/                # Main pipeline: features → profile → text labels
│   ├── core/                 # orchestration + profile assembly
│   ├── models/               # OpenSMILE features + Gender/Accent classifiers
│   ├── utils/                # audio IO, composite metrics, checkpoints
│   └── scripts/              # profile → text labels (quantile binning)
├── dataspeech/               # pitch / SNR / speaking-rate enrichments
│   ├── gpu_enrichments/      # penn (F0) + brouhaha (SNR/C50) — auto CPU/GPU
│   └── cpu_enrichments/      # speaking rate (Vietnamese syllables/sec)
├── prompts/                  # natural-language descriptions (Gemini)
├── alignment/                # forced alignment (stable-ts + Whisper) — separate env
├── requirements.txt         
├── requirements_cpu.txt     
├── requirements_gpu.txt      
└── README.md
```

## Age & Emotion

Age and emotion estimation were dropped from the active pipeline (they were the
heaviest models and are optional).

- **Age** — `audeering/wav2vec2-large-robust-24-ft-age-gender`: a wav2vec2 model
  regressing speaker age (and gender) directly from the raw waveform.
- **Emotion** — `MERaLiON/MERaLiON-SER-v1`: a Whisper-backbone speech-emotion
  recognition classifier (neutral / happy / angry / sad / ...).

