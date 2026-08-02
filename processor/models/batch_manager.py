import librosa
from unified_analysis.config import SAMPLING_RATE
from .ai_classifiers import GenderClassifier, AccentClassifier

class BatchAIClassifier:
    def __init__(self):
        self.gender_classifier = GenderClassifier()
        self.accent_classifier = AccentClassifier()

    def predict_batch_from_cache(self, filepaths: list, cache) -> list:
        # Collect all audio arrays
        audio_arrays = []
        for filepath in filepaths:
            if cache is not None:
                audio = cache.get(filepath, SAMPLING_RATE)
                if audio is None:
                    # Cache miss - load from file
                    audio, _ = librosa.load(filepath, sr=SAMPLING_RATE)
            else:
                audio, _ = librosa.load(filepath, sr=SAMPLING_RATE)
            audio_arrays.append(audio)
        
        # Batch predict
        genders = self.gender_classifier.predict_batch(audio_arrays, SAMPLING_RATE)
        accents = self.accent_classifier.predict_batch(audio_arrays, SAMPLING_RATE)

        # Combine results
        results = []
        for i, filepath in enumerate(filepaths):
            results.append({
                "filepath": filepath,
                "gender": genders[i],
                "accent": accents[i],
            })
        
        return results
    
    def predict_batch(self, filepaths: list) -> list:
        return self.predict_batch_from_cache(filepaths, None)

