import os
import glob
import librosa
import soundfile as sf
import numpy as np
from unified_analysis.config import SAMPLING_RATE, MAX_AUDIO_DURATION, AUDIO_EXTENSIONS

def get_audio_files(audio_dir):
    # streaming file
    for root, _, files in os.walk(audio_dir):
        for file in files:
            # Check extension
            if any(file.lower().endswith(ext.replace("*", "")) for ext in AUDIO_EXTENSIONS):
                yield os.path.join(root, file)

def load_audio_16k(path, max_sec=MAX_AUDIO_DURATION):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Ensure mono
    if len(audio.shape) != 1:
        audio = audio.flatten()

    
    if sr != SAMPLING_RATE:
        min_samples = 2200
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
             
    max_len = int(max_sec * SAMPLING_RATE)
    if len(audio) > max_len:
        audio = audio[:max_len]
    return audio

def audio_duration_sec(path):
    """Duration of an audio file in seconds (reads header only, no full decode)."""
    info = sf.info(path)
    return info.frames / info.samplerate

class AudioCache:
    def __init__(self):
        self._cache = {}  # {filepath: {sr: audio_array}}
    
    def get(self, filepath, target_sr=SAMPLING_RATE):
        if filepath not in self._cache:
            self._cache[filepath] = {}
        
        if target_sr in self._cache[filepath]:
            return self._cache[filepath][target_sr]
        
        if self._cache[filepath]:
            # Resample from cached version
            cached_sr = next(iter(self._cache[filepath].keys()))
            cached_audio = self._cache[filepath][cached_sr]
            if cached_sr != target_sr:
                resampled = librosa.resample(cached_audio, orig_sr=cached_sr, target_sr=target_sr)
                self._cache[filepath][target_sr] = resampled
                return resampled
            return cached_audio
        
        # Load data
        audio, _ = librosa.load(filepath, sr=target_sr)
        self._cache[filepath][target_sr] = audio
        return audio
    
    def preload(self, filepath, audio_array, sample_rate):
        if filepath not in self._cache:
            self._cache[filepath] = {}
        self._cache[filepath][sample_rate] = audio_array
    
    def clear(self):
        self._cache.clear()
    
    def __len__(self):
        return len(self._cache)
