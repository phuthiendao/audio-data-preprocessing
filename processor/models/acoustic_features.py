# acoustic_features.py
import opensmile
import librosa
import numpy as np

class OpenSmileAcousticFeatures:    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        try:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        except Exception as e:
            print(f"Error initializing OpenSMILE: {e}")
            self.smile = None

    def compute(self, audio_path: str, audio_array=None) -> dict:
        if self.smile is None:
            return {}

        try:
            # Load audio if not provided
            if audio_array is None:
                y, sr = librosa.load(audio_path, sr=self.sample_rate)
            else:
                y = audio_array
            
            df = self.smile.process_signal(y, self.sample_rate)

            # Speech duration
            speech_duration = len(y) / self.sample_rate
            
            features = {
                # Speech Duration
                "speech_duration": speech_duration,
                
                # Energy / Loudness
                "loudness": float(df["loudness_sma3_amean"].iloc[0]),
                "loudness_std": float(df["loudness_sma3_stddevNorm"].iloc[0]),

                # Voice quality
                "jitter": float(df["jitterLocal_sma3nz_amean"].iloc[0]),
                "shimmer": float(df["shimmerLocaldB_sma3nz_amean"].iloc[0]),

                # Formants
                "formant_f1": float(df["F1frequency_sma3nz_amean"].iloc[0]),
                "formant_f2": float(df["F2frequency_sma3nz_amean"].iloc[0]),
                
                # HNR (breathiness)
                "hnr": float(df["HNRdBACF_sma3nz_amean"].iloc[0]),
                
                # Spectral flux
                "spectral_flux": float(df["spectralFlux_sma3_amean"].iloc[0]),

                # Resonance
                "alpha_ratio": float(df["alphaRatioV_sma3nz_amean"].iloc[0]),

                # Breathiness
                "hammarberg": float(df["hammarbergIndexV_sma3nz_amean"].iloc[0]),

                # Spectral Slopes
                "spectral_slope_0_500": float(df["slopeV0-500_sma3nz_amean"].iloc[0]),
                "spectral_slope_500_1500": float(df["slopeV500-1500_sma3nz_amean"].iloc[0]),
            }

            # Spectral centroid from Librosa
            try:
                centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)
                features["spectral_centroid"] = float(np.mean(centroid))
            except Exception as e:
                print(f"Librosa spectral_centroid failed: {e}")
                features["spectral_centroid"] = None

            return features

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return {}
