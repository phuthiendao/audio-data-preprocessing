# config.py

SAMPLING_RATE = 16000
MIN_AUDIO_DURATION = 1.0   # drop audio shorter than this (seconds)
MAX_AUDIO_DURATION = 30.0  # drop audio longer than this (seconds)
AUDIO_EXTENSIONS = ["*.wav", "*.mp3", "*.m4a"]

CHECKPOINT_INTERVAL = 30  # save checkpoint every 30 seconds
MEMORY_CLEANUP_INTERVAL = 50
MAX_RETRIES = 2
DECIMAL_PLACES = 4

# OpenSMILE feature ranges: group 1 (voice quality/formants/pitch) = literature-based
# population ranges; group 2 (eGeMAPS spectral) = wide physical bounds (refine via corpus z-score).
# Non-OpenSMILE keys (pitch_std=penn, snr/c50=brouhaha, speech_rate=g2p) left unchanged.
NORM_RANGES = {
    # Voice Quality
    "hnr": (0.0, 18.0),           
    "jitter": (0.0, 0.05),      
    "shimmer": (0.0, 2.0),     
    "hammarberg": (0.0, 40.0),    
    "alpha_ratio": (-40.0, 5.0), 
    
    # Formants
    "formant_f1": (250, 1100),     
    "formant_f2": (700, 2900),     
    
    # Spectral / Timbre
    "spectral_centroid": (300, 3500),    
    "spectral_flux": (0.0, 1.5),       
    "spectral_slope_0_500": (-0.05, 0.10),   
    "spectral_slope_500_1500": (-0.05, 0.02), 
    
    # Prosody
    "loudness": (0.0, 4.0),      
    "loudness_std": (0.0, 1.5),  
    
    # Pitch Stats (single source: penn FCNF0++)
    "pitch_std": (10.0, 70.0),   # Hz - voice.pitch_std output only (unused by composites)
    "pitch_std_st": (0.0, 8.0),  # semitones - penn F0 std -> intonation/expressiveness
    "pitch_range": (0.0, 20.0),  # semitones - penn F0 p80-p20
    
    # 
    "speech_duration": (0.5, 20.0),
    "speech_rate": (2.0, 9.0),  # syllables/sec (Vietnamese)
    "snr": (15, 80),
    "c50": (0, 60),
}


# Text Mapping Bins 
TEXT_MAPPING_BINS = {
    # Speaking Rate 
    "speed": {
        "edges": [0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0],
        "labels": ["very slowly", "slowly", "slightly slowly", "moderate speed", "slightly fast", "fast", "very fast"]
    },
    
    # Noise (SNR)
    "snr": {
        "edges": [17.1, 25.4, 33.7, 41.9, 50.2, 58.5, 66.8, 75.0], 
        "labels": ["very noisy", "noisy", "slightly noisy", "balanced in clarity", "slightly clean", "clean", "very clean"]
    },
    
    # Reverberation (C50)
    "reverb": {
        "edges": [0, 40, 50, 55, 58, 65],
        "labels": ["very distant-sounding", "distant-sounding", "slightly distant-sounding", "slightly close-sounding", "very close-sounding"]
    },
    
    # Speech Monotony (Gender Specific)
    "monotony_male": {
        "edges": [0.0, 20.0, 30.0, 40.0, 55.0, 150.0],
        "labels": ["very monotone", "monotone", "slightly expressive and animated", "expressive and animated", "very expressive and animated"]
    },
    "monotony_female": {
        "edges": [0.0, 30.0, 40.0, 50.0, 70.0, 150.0],
        "labels": ["very monotone", "monotone", "slightly expressive and animated", "expressive and animated", "very expressive and animated"]
    },
    
    # Pitch (Gender specific)
    "pitch_male": {
        "edges": [64.7, 81.7, 98.7, 115.7, 132.7, 149.7, 166.7, 183.7],
        "labels": ["very low-pitch", "low-pitch", "slightly low-pitch", "moderate pitch", "slightly high-pitch", "high-pitch", "very high-pitch"]
    },
    "pitch_female": {
        "edges": [120.2, 141.6, 163.1, 184.5, 206.0, 227.4, 248.9, 270.3],
        "labels": ["very low-pitch", "low-pitch", "slightly low-pitch", "moderate pitch", "slightly high-pitch", "high-pitch", "very high-pitch"]
    }
}

# Composite Score Bins
COMPOSITE_SCORE_BINS = {
    # Timbre (Median 0.53)
    "timbre": {
        "ranges": [(0, 0.38), (0.38, 0.48), (0.48, 0.62), (0.62, 1.0)],
        "labels": ["dark", "warm", "neutral", "bright"]
    },
    # Breathiness (Median 0.48)
    "breathiness": {
        "ranges": [(0, 0.55), (0.55, 0.65), (0.65, 0.80), (0.80, 1.0)],
        "labels": ["clear", "slightly breathy", "breathy", "very breathy"]
    },
    # Resonance (Median 0.53)
    "resonance": {
        "ranges": [(0, 0.35), (0.35, 0.55), (0.55, 0.70), (0.70, 1.0)],
        "labels": ["chest voice", "balanced", "forward", "nasal"]
    },
    # Energy (Median 0.29)
    "energy": {
        "ranges": [(0, 0.20), (0.20, 0.45), (0.45, 0.70), (0.70, 1.0)],
        "labels": ["soft", "moderate", "energetic", "powerful"]
    },
    # Stability (Median 0.67)
    "stability": {
        "ranges": [(0, 0.45), (0.45, 0.60), (0.60, 0.80), (0.80, 1.0)],
        "labels": ["unstable", "slightly unstable", "stable", "very stable"]
    },
    # Expressiveness (Median 0.38)
    "expressiveness": {
        "ranges": [(0, 0.25), (0.25, 0.45), (0.45, 0.65), (0.65, 1.0)],
        "labels": ["flat", "slightly expressive", "expressive", "very expressive"]
    },
    # Intonation (Median 0.33)
    "intonation": {
        "ranges": [(0, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 1.0)],
        "labels": ["very monotone", "monotone", "expressive", "very expressive"]
    },
    # Clarity (Median 0.83)
    "clarity": {
        "ranges": [(0, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 1.0)],
        "labels": ["muffled", "slightly clear", "clear", "very clear"]
    }
}
