import os
import sys
import argparse
import json

import numpy as np

from unified_analysis.config import TEXT_MAPPING_BINS, COMPOSITE_SCORE_BINS

# Bins are computed from the dataset distribution at convert time. Below this
# many valid samples the distribution is too small, so we fall back to config.
MIN_N = 50

# Binning method (how score -> label boundaries are placed):
#   "gaussian": equal-width over [mean - K*std, mean + K*std] -> BELL-shaped label
#               counts (most samples in middle labels, few at the extremes). Default.
#   "quantile": equal-frequency -> flat/uniform label counts.
#   "width":    equal-width over the observed [min, max].
BIN_METHOD = "gaussian"
BIN_STD_SPAN = 2.5  # gaussian only: half-width of the binned range, in std units

# Each binned label: where it goes in the output, which profile value feeds it,
# the label vocabulary, and how to fall back when data is insufficient.
#   kind="composite" -> value is a 0..1 score, fallback = COMPOSITE_SCORE_BINS[cfg]
#   kind="text"      -> value is a raw metric, fallback = TEXT_MAPPING_BINS[cfg]
#   gender=True      -> edges computed per gender; fallback = TEXT_MAPPING_BINS[cfg_<gender>]
BIN_SPECS = [
    {"out": ("voice", "pitch"),                 "val": ("voice", "pitch_hz"),          "kind": "text",      "cfg": "pitch",   "gender": True},
    {"out": ("voice", "timbre"),                "val": ("voice", "timbre"),            "kind": "composite", "cfg": "timbre"},
    {"out": ("voice", "breathiness"),           "val": ("voice", "breathiness"),       "kind": "composite", "cfg": "breathiness"},
    {"out": ("voice", "resonance"),             "val": ("voice", "resonance"),         "kind": "composite", "cfg": "resonance"},
    {"out": ("speech", "speed"),                "val": ("speech", "speed_wps"),        "kind": "text",      "cfg": "speed"},
    {"out": ("speech", "clarity"),              "val": ("speech", "clarity"),          "kind": "composite", "cfg": "clarity"},
    {"out": ("speech", "intonation"),           "val": ("speech", "intonation"),       "kind": "composite", "cfg": "intonation"},
    {"out": ("expression", "energy"),           "val": ("expression", "energy"),       "kind": "composite", "cfg": "energy"},
    {"out": ("expression", "expressiveness"),   "val": ("expression", "expressiveness"),"kind": "composite","cfg": "expressiveness"},
    {"out": ("technical", "recording_quality"), "val": ("technical", "snr_db"),        "kind": "text",      "cfg": "snr"},
    {"out": ("technical", "reverberation"),     "val": ("technical", "c50_db"),        "kind": "text",      "cfg": "reverb"},
    {"out": ("technical", "stability"),         "val": ("technical", "stability"),     "kind": "composite", "cfg": "stability"},
]


class ProfileToTextMapper:
    def __init__(self):
        # Data-driven internal edges per output field. Value is a list of edges
        # (len == n_labels - 1), or {"male": edges, "female": edges} for gender
        # fields, or None to signal fallback to config.
        self.edges = {}

    # ---------- helpers ----------
    @staticmethod
    def _key(spec):
        return "/".join(spec["out"])

    def _labels(self, spec):
        if spec["kind"] == "composite":
            return COMPOSITE_SCORE_BINS[spec["cfg"]]["labels"]
        if spec.get("gender"):
            return TEXT_MAPPING_BINS[spec["cfg"] + "_male"]["labels"]
        return TEXT_MAPPING_BINS[spec["cfg"]]["labels"]

    @staticmethod
    def _get_val(profile, spec):
        sec, k = spec["val"]
        return profile.get(sec, {}).get(k)

    @staticmethod
    def _clean(values):
        return np.array(
            [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
            dtype=float,
        )

    @classmethod
    def _compute_edges(cls, values, n_labels):
        """Internal label edges (n_labels - 1 of them) per BIN_METHOD, or None if too few samples."""
        vals = cls._clean(values)
        if len(vals) < MIN_N:
            return None, len(vals)
        if BIN_METHOD == "quantile":
            qs = np.linspace(0.0, 1.0, n_labels + 1)[1:-1]
            return list(np.quantile(vals, qs)), len(vals)
        if BIN_METHOD == "width":
            lo, hi = float(np.min(vals)), float(np.max(vals))
        else:  # "gaussian": mean +/- K*std -> bell-shaped label counts
            mu, sigma = float(np.mean(vals)), float(np.std(vals))
            if sigma == 0:
                return None, len(vals)
            lo, hi = mu - BIN_STD_SPAN * sigma, mu + BIN_STD_SPAN * sigma
        return list(np.linspace(lo, hi, n_labels + 1)[1:-1]), len(vals)

    # ---------- fallbacks (config-based, legacy behaviour) ----------
    def _get_bin_label(self, value, edges, labels):
        if value is None:
            return "undefined"
        if value < edges[0]:
            return labels[0]
        if value >= edges[-1]:
            return labels[-1]
        for i in range(len(edges) - 1):
            if edges[i] <= value < edges[i + 1]:
                return labels[i]
        return labels[-1]

    def _score_to_label(self, score, bins):
        if score is None:
            return "undefined"
        for (lo, hi), label in zip(bins["ranges"], bins["labels"]):
            if lo <= score < hi:
                return label
        return bins["labels"][-1]

    @staticmethod
    def _label_from_edges(value, edges, labels):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "undefined"
        idx = int(np.searchsorted(edges, value, side="right"))
        return labels[idx]

    # ---------- pass 1: fit edges from the dataset ----------
    def fit(self, profiles):
        for spec in BIN_SPECS:
            key = self._key(spec)
            n_labels = len(self._labels(spec))
            if spec.get("gender"):
                self.edges[key] = {}
                for g in ("male", "female"):
                    vals = [self._get_val(p, spec) for p in profiles
                            if p.get("identity", {}).get("gender") == g]
                    e, n = self._compute_edges(vals, n_labels)
                    self.edges[key][g] = e
                    if e is None:
                        print(f"[warn] {key}/{g}: N={n} < {MIN_N}, fallback to config edges")
            else:
                vals = [self._get_val(p, spec) for p in profiles]
                e, n = self._compute_edges(vals, n_labels)
                self.edges[key] = e
                if e is None:
                    print(f"[warn] {key}: N={n} < {MIN_N}, fallback to config edges")

    # ---------- pass 2: assign a label for one field ----------
    def _label_field(self, profile, spec):
        key = self._key(spec)
        labels = self._labels(spec)
        value = self._get_val(profile, spec)

        if spec.get("gender"):
            g = profile.get("identity", {}).get("gender")
            e = self.edges.get(key, {}).get(g) if g in ("male", "female") else None
            if e is not None:
                return self._label_from_edges(value, e, labels)
            if g in ("male", "female"):
                cfg = TEXT_MAPPING_BINS[spec["cfg"] + "_" + g]
                return self._get_bin_label(value, cfg["edges"], cfg["labels"])
            return "undefined"

        e = self.edges.get(key)
        if e is not None:
            return self._label_from_edges(value, e, labels)
        if spec["kind"] == "composite":
            return self._score_to_label(value, COMPOSITE_SCORE_BINS[spec["cfg"]])
        cfg = TEXT_MAPPING_BINS[spec["cfg"]]
        return self._get_bin_label(value, cfg["edges"], cfg["labels"])

    def profile_to_text(self, profile: dict) -> dict:
        identity = profile.get("identity", {})
        out = {
            "identity": {
                "gender": identity.get("gender", "undefined"),
                "age_range": identity.get("age_range", "undefined"),
                "region_accent": identity.get("region_accent", "undefined"),
            },
            "voice": {},
            "speech": {},
            "expression": {"emotion": profile.get("expression", {}).get("emotion", "undefined")},
            "technical": {},
        }
        for spec in BIN_SPECS:
            sec, k = spec["out"]
            out[sec][k] = self._label_field(profile, spec)
        return out

    # ---------- io ----------
    @staticmethod
    def _iter_profiles(input_file):
        if input_file.endswith(".jsonl"):
            with open(input_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception as e:
                        print(f"Error parsing line: {e}")
        else:
            with open(input_file, "r", encoding="utf-8") as f_in:
                for profile in json.load(f_in):
                    yield profile

    def _dump_edges(self, output_path):
        bins_path = output_path.rsplit(".", 1)[0] + "_bins.json"
        try:
            with open(bins_path, "w", encoding="utf-8") as f:
                json.dump({"min_n": MIN_N, "edges": self.edges}, f, ensure_ascii=False, indent=2)
            print(f"Data-driven bin edges saved to: {bins_path}")
        except Exception as e:
            print(f"Could not save bin edges: {e}")

    def process(self, input_file, output_file=None, audio_dir=None):
        output_path = output_file or input_file.replace(".jsonl", "").replace(".json", "") + "_text.jsonl"
        print(f"Converting {input_file} -> {output_path} ...")

        # Pass 1: fit quantile edges from the whole dataset.
        self.fit(list(self._iter_profiles(input_file)))
        self._dump_edges(output_path)

        # Pass 2: assign labels and write.
        count = 0
        try:
            with open(output_path, "w", encoding="utf-8") as f_out:
                for profile in self._iter_profiles(input_file):
                    fname = profile.get("filename")
                    if audio_dir:
                        fname = os.path.join(audio_dir, fname)
                    record = {"filename": fname}
                    record.update(self.profile_to_text(profile))
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 1000 == 0:
                        print(f"Converted {count}...", end="\r")
        except Exception as e:
            print(f"Conversion failed: {e}")
            return

        print(f"\nCompleted! Processed {count} records.")
        print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Voice Profile to text")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default=None, help="Root directory to prepend to filenames")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"File not found: {args.input_file}")
        return

    mapper = ProfileToTextMapper()
    mapper.process(args.input_file, args.output_file, args.audio_dir)


if __name__ == "__main__":
    main()
