from processor.config import DECIMAL_PLACES
from processor.utils.metrics import (
    compute_breathiness, compute_timbre, compute_resonance, compute_energy,
    compute_stability, compute_intonation, compute_expressiveness, compute_clarity
)

import math

class ProfileBuilder:
    def __init__(self):
        pass

    def _safe_round(self, value, decimals):
        if value is None:
            return None
        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return None
            return round(val, decimals)
        except (ValueError, TypeError):
            return None

    def build(self, data, ai_result):
        acoustic = data.get("acoustic", {})
        
        gender = ai_result["gender"]
        accent = ai_result["accent"]
        emotion = ai_result.get("emotion", "undefined")
        
        pitch_mean = data.get("pitch_mean", 0)
        pitch_std = data.get("pitch_std", 0)                # Hz (voice.pitch_std output)
        pitch_std_st = data.get("pitch_std_st", 0)          # semitones (penn F0 std)
        pitch_range_st = data.get("pitch_range_st")         # semitones (penn F0 p80-p20)
        snr = data.get("snr", 0)
        
        age_range = ai_result.get("age_range", "unknown")
        
        # Composite scores
        breathiness = compute_breathiness(
            acoustic.get("hnr"), 
            acoustic.get("jitter"), 
            acoustic.get("shimmer"),
            acoustic.get("hammarberg")
        )
        timbre = compute_timbre(
            acoustic.get("spectral_centroid"),
            acoustic.get("spectral_slope_0_500"),
            acoustic.get("spectral_slope_500_1500")
        )
        resonance = compute_resonance(
            acoustic.get("formant_f1"), 
            acoustic.get("formant_f2"),
            acoustic.get("alpha_ratio")
        )
        energy = compute_energy(
            acoustic.get("loudness"), 
            acoustic.get("spectral_flux")
        )
        stability = compute_stability(
            acoustic.get("jitter"), 
            acoustic.get("shimmer")
        )
        intonation = compute_intonation(pitch_std_st, pitch_range_st)
        expressiveness = compute_expressiveness(
            pitch_std_st,
            acoustic.get("loudness_std", 0),
            data.get("speaking_rate")
        )
        clarity = compute_clarity(snr, data.get("c50"))
        
        return {
            "filename": data["filename"],
            "identity": {
                "gender": gender,
                "age_range": age_range,
                "region_accent": accent,
            },
            "voice": {
                "pitch_hz": self._safe_round(pitch_mean, DECIMAL_PLACES),
                "pitch_std": self._safe_round(pitch_std, DECIMAL_PLACES),
                "timbre": timbre,
                "breathiness": breathiness,
                "resonance": resonance,
            },
            "speech": {
                "speed_wps": round(data.get("speaking_rate") or 0, DECIMAL_PLACES) if data.get("speaking_rate") else None,
                "clarity": clarity,
                "intonation": intonation,
            },
            "expression": {
                "emotion": emotion,
                "energy": energy,
                "expressiveness": expressiveness,
            },
            "technical": {
                "snr_db": self._safe_round(snr, DECIMAL_PLACES),
                "c50_db": self._safe_round(data.get("c50"), DECIMAL_PLACES),
                "stability": stability,
            },
        }
