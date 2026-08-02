# ai_classifiers.py
import torch
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class BaseAudioClassifier:    
    model_id = ""  # Override in subclass
    
    def __init__(self):
        self.device = get_device()
        self.model = None
        self.feature_extractor = None
        
        if self.model_id:
            try:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_id).to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Error loading {self.__class__.__name__}: {e}")
                self.model = None
    
    def _preprocess_audio(self, audio_array, sampling_rate):
        target_sr = self.feature_extractor.sampling_rate
        if sampling_rate != target_sr:
            return librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=target_sr)
        return audio_array
    
    def _load_audio(self, audio_path):
        target_sr = self.feature_extractor.sampling_rate
        audio_array, _ = librosa.load(audio_path, sr=target_sr)
        return audio_array
    
    def _extract_features(self, audio_input, sampling_rate):
        return self.feature_extractor(audio_input, sampling_rate=sampling_rate, 
                                      return_tensors="pt", padding=True)
    
    def _extract_features_batch(self, audio_list, sampling_rate):
        return self.feature_extractor(audio_list, sampling_rate=sampling_rate,
                                      return_tensors="pt", padding=True)
    
    def _postprocess_label(self, label):
        return label
    
    def predict(self, audio_path, audio_array=None, sampling_rate=None):
        if not self.model:
            return "unknown"
        
        try:
            target_sr = self.feature_extractor.sampling_rate
            
            # Load or preprocess audio
            if audio_array is None:
                audio_array = self._load_audio(audio_path)
            elif sampling_rate is not None:
                audio_array = self._preprocess_audio(audio_array, sampling_rate)
            
            # Extract features
            inputs = self._extract_features(audio_array, target_sr)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Get prediction
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            predicted_label = self.model.config.id2label[predicted_class_id]
            
            return self._postprocess_label(predicted_label)
        except Exception as e:
            print(f"Error predicting for {audio_path}: {e}")
            return "error"
    
    def predict_batch(self, audio_arrays: list, sampling_rate: int = 16000) -> list:
        """Batch prediction for multiple audio samples - true GPU batching."""
        if not self.model:
            return ["unknown"] * len(audio_arrays)
        
        # Validate input
        if not audio_arrays:
            return []
        
        try:
            target_sr = self.feature_extractor.sampling_rate
            
            # Resample all to target SR if needed
            resampled = [self._preprocess_audio(audio, sampling_rate) for audio in audio_arrays]
            
            # Extract features as batch
            inputs = self._extract_features_batch(resampled, target_sr)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Single batched inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Get predictions
            predicted_ids = torch.argmax(logits, dim=-1).tolist()
            labels = [self.model.config.id2label[id] for id in predicted_ids]
            
            return [self._postprocess_label(label) for label in labels]
        except Exception as e:
            print(f"Batch prediction failed, retrying individually: {e}")
            # Fallback: predict each sample individually
            results = []
            for audio in audio_arrays:
                try:
                    result = self.predict(None, audio_array=audio, sampling_rate=sampling_rate)
                    results.append(result)
                except:
                    results.append("error")
            return results


class GenderClassifier(BaseAudioClassifier):    
    def __init__(self):
        self.model_id = "thangtrungnguyen/vietnamese-female-male-voice-classification-model"
        super().__init__()
    
    def _postprocess_label(self, label):
        return label.lower()


class AccentClassifier(BaseAudioClassifier):    
    def __init__(self):
        self.model_id = "thangtrungnguyen/vietnamese-regional-accent-classification-model"
        super().__init__()


class EmotionClassifier(BaseAudioClassifier):    
    def __init__(self):
        self.model_id = "MERaLiON/MERaLiON-SER-v1"
        self.device = get_device()
        print(f"Loading Emotion Classifier ({self.model_id}) on device...")
        
        # Monkey-patch torch.logspace to fix meta-tensor error in model init
        original_logspace = torch.logspace
        def patched_logspace(*args, **kwargs):
            if 'device' not in kwargs:
                kwargs['device'] = 'cpu'
            return original_logspace(*args, **kwargs)
        torch.logspace = patched_logspace
        
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_id, trust_remote_code=True).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading EmotionClassifier: {e}")
            self.model = None
        finally:
            torch.logspace = original_logspace
    
    def _extract_features(self, audio_input, sampling_rate):
        """Whisper-specific: fixed 30s input (480,000 samples at 16kHz)."""
        # MERaLiON uses Whisper backbone, needs same padding
        return self.feature_extractor(
            audio_input, 
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=480000,
            truncation=True
        )
    
    def _extract_features_batch(self, audio_list, sampling_rate):
        """Whisper-specific batch processing."""
        return self.feature_extractor(
            audio_list,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=480000,
            truncation=True
        )

    def predict(self, audio_path, audio_array=None, sampling_rate=None):
        """Predict single audio sample with dict output handling."""
        if not self.model:
            return "unknown"
        
        try:
            target_sr = self.feature_extractor.sampling_rate
            
            if audio_array is None:
                audio_array = self._load_audio(audio_path)
            elif sampling_rate is not None:
                audio_array = self._preprocess_audio(audio_array, sampling_rate)
            
            inputs = self._extract_features(audio_array, target_sr)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Handle dict output from MERaLiON
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            # Handle possible string keys in id2label
            predicted_label = self.model.config.id2label.get(predicted_class_id) or self.model.config.id2label.get(str(predicted_class_id))
            
            return self._postprocess_label(predicted_label)
        except Exception as e:
            print(f"Error predicting for {audio_path}: {e}")
            return "error"

    def predict_batch(self, audio_arrays: list, sampling_rate: int = 16000) -> list:
        """Batch prediction with dict output handling."""
        if not self.model:
            return ["unknown"] * len(audio_arrays)
        
        if not audio_arrays:
            return []
        
        try:
            target_sr = self.feature_extractor.sampling_rate
            resampled = [self._preprocess_audio(audio, sampling_rate) for audio in audio_arrays]
            
            inputs = self._extract_features_batch(resampled, target_sr)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            
            predicted_ids = torch.argmax(logits, dim=-1).tolist()
            labels = []
            for pid in predicted_ids:
                label = self.model.config.id2label.get(pid) or self.model.config.id2label.get(str(pid))
                labels.append(label)
            
            return [self._postprocess_label(label) for label in labels]
        except Exception as e:
            print(f"EmotionClassifier batch failed, retrying individually: {e}")
            # Fallback: predict each sample individually
            results = []
            for audio in audio_arrays:
                try:
                    result = self.predict(None, audio_array=audio, sampling_rate=sampling_rate)
                    results.append(result)
                except:
                    results.append("error")
            return results
