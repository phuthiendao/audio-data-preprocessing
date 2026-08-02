import numpy as np
from unified_analysis.config import NORM_RANGES, DECIMAL_PLACES


def normalize(value, min_val, max_val):
    if value is None or np.isnan(value):
        return 0.5
    # Guard against division by zero
    if max_val == min_val:
        return 0.5
    return float(np.clip((value - min_val) / (max_val - min_val), 0, 1))


def compute_breathiness(hnr, jitter, shimmer, hammarberg=None):
    """HNR (Low HNR = Breathy), Jitter (High = Breathy), Shimmer (High = Breathy), Hammarberg (High = Breathy)"""
    hnr_norm = normalize(hnr, *NORM_RANGES["hnr"])
    jitter_norm = normalize(jitter, *NORM_RANGES["jitter"])
    shimmer_norm = normalize(shimmer, *NORM_RANGES["shimmer"])
    
    if hammarberg is not None:
        # With hammarberg
        w1, w2, w3, w4 = 0.5, 0.1, 0.1, 0.3
        hammarberg_norm = normalize(hammarberg, *NORM_RANGES["hammarberg"])
        score = w1 * (1 - hnr_norm) + w2 * jitter_norm + w3 * shimmer_norm + w4 * hammarberg_norm
    else:
        # Without hammarberg
        w1, w2, w3 = 0.714, 0.143, 0.143    
        score = w1 * (1 - hnr_norm) + w2 * jitter_norm + w3 * shimmer_norm
    return round(score, DECIMAL_PLACES)


def compute_timbre(spectral_centroid, slope_0_500=None, slope_500_1500=None):
    """Timbre: Integrated Brightness + Fullness High Centroid = Bright/Thin + High Slope 0-500 = Full/Deep"""
    
    centroid_norm = normalize(spectral_centroid, *NORM_RANGES["spectral_centroid"]) # Brightness
    
    if slope_0_500 is not None and slope_500_1500 is not None:
        norm_slope1 = normalize(slope_0_500, *NORM_RANGES["spectral_slope_0_500"])
        norm_slope2 = normalize(slope_500_1500, *NORM_RANGES["spectral_slope_500_1500"])
        w1, w2, w3 = 0.4, 0.3, 0.3
        score = w1 * centroid_norm + w2 * norm_slope1 + w3 * norm_slope2
        return round(score, DECIMAL_PLACES)
    return round(centroid_norm, DECIMAL_PLACES)


def compute_resonance(f1, f2, alpha_ratio=None):
    """Resonance: High Score = Nasal/Head (Thin). Low Score = Chest (Full)."""
    if f1 is None or f2 is None or f1 == 0:
        return 0.5
        
    # Valid Formant Ratio
    ratio = f2 / f1
    ratio_norm = normalize(ratio, 1.5, 3.5)
    
    if alpha_ratio is not None:
        alpha_norm = normalize(alpha_ratio, *NORM_RANGES["alpha_ratio"])
        w1, w2 = 0.6, 0.4
        score = w1 * ratio_norm + w2 * (1 - alpha_norm)
        return round(score, DECIMAL_PLACES)
    
    return round(ratio_norm, DECIMAL_PLACES)


def compute_energy(loudness, spectral_flux):
    """Energy: Loudness + Spectral Flux"""
    w1, w2 = 0.65, 0.35
    loud_norm = normalize(loudness, *NORM_RANGES["loudness"])
    flux_norm = normalize(spectral_flux, *NORM_RANGES["spectral_flux"])
    return round(w1 * loud_norm + w2 * flux_norm, DECIMAL_PLACES)


def compute_stability(jitter, shimmer):
    """Stability: Jitter + Shimmer"""
    w1, w2 = 0.5, 0.5
    jitter_norm = normalize(jitter, *NORM_RANGES["jitter"])
    shimmer_norm = normalize(shimmer, *NORM_RANGES["shimmer"])
    return round(1 - (w1 * jitter_norm + w2 * shimmer_norm), DECIMAL_PLACES)


def compute_intonation(pitch_std_st, pitch_range_st=None):
    """Intonation from a single F0 source (penn), in semitones: variation (std) + range (p80-p20)."""
    std_norm = normalize(pitch_std_st, *NORM_RANGES["pitch_std_st"])

    if pitch_range_st is not None:
        range_norm = normalize(pitch_range_st, *NORM_RANGES["pitch_range"])

        w1, w2 = 0.4, 0.6
        score = w1 * std_norm + w2 * range_norm
        return round(score, DECIMAL_PLACES)
    return round(std_norm, DECIMAL_PLACES)


def compute_expressiveness(pitch_std_st, loudness_std, speech_rate=None):
    """Compute expressiveness from pitch variation (penn, semitones) and loudness variation."""
    pitch_norm = normalize(pitch_std_st, *NORM_RANGES["pitch_std_st"])
    loud_norm = normalize(loudness_std, *NORM_RANGES["loudness_std"])
    
    if speech_rate is not None:
        rate_norm = normalize(speech_rate, *NORM_RANGES["speech_rate"])
        w1, w2, w3 = 0.5, 0.3, 0.2
        score = w1 * pitch_norm + w2 * loud_norm + w3 * rate_norm
        return round(score, DECIMAL_PLACES)

    w1, w2 = 0.6, 0.4
    score = w1 * pitch_norm + w2 * loud_norm
    return round(score, DECIMAL_PLACES)


def compute_clarity(snr, c50=None):
    """Compute clarity from SNR and optional C50.
    Returns None if SNR is None/NaN (cannot estimate clarity without SNR)."""
    
    # Critical: If SNR is null/NaN, we cannot compute clarity -> return None
    if snr is None or (isinstance(snr, float) and np.isnan(snr)):
        return None
    
    snr_norm = normalize(snr, *NORM_RANGES["snr"])
    
    if c50 is not None and not (isinstance(c50, float) and np.isnan(c50)):
        c50_norm = normalize(c50, *NORM_RANGES["c50"])
        w1, w2 = 0.6, 0.4
        score = w1 * snr_norm + w2 * c50_norm
        return round(score, DECIMAL_PLACES)
    
    return round(snr_norm, DECIMAL_PLACES)
