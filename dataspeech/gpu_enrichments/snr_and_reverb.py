from pyannote.audio import Model
from pathlib import Path
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
import torch 
from huggingface_hub import hf_hub_download
import numpy as np

_model_cache = {}
_pipeline_cache = {}
ratio = 16000/270


def _get_pipeline(rank=None, batch_size=32):
    global _model_cache, _pipeline_cache
    
    cache_key = (rank or 0, batch_size)
    
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]
    
    # Load model if not cached
    if "model" not in _model_cache:
        _model_cache["model"] = Model.from_pretrained(
            Path(hf_hub_download(repo_id="ylacombe/brouhaha-best", filename="best.ckpt")),
            strict=False,
        )
    
    model = _model_cache["model"]
    
    # Move to GPU if available
    if rank is not None or torch.cuda.device_count() > 0:
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        model.to(device)
    
    # Create and cache pipeline
    pipeline = RegressiveActivityDetectionPipeline(segmentation=model, batch_size=batch_size)
    if rank is not None:
        pipeline.to(torch.device(device))
    
    _pipeline_cache[cache_key] = pipeline
    return pipeline


def snr_apply(batch, rank=None, audio_column_name="audio", batch_size=64):
    """
    Apply SNR and C50 estimation with CACHED pipeline.
    Much faster than original version that recreated pipeline each call.
    """
    # Get cached pipeline (OPTIMIZED - no recreation overhead)
    pipeline = _get_pipeline(rank, batch_size)
    device = pipeline._models["segmentation"].device

    if isinstance(batch[audio_column_name], list):  
        snr = []
        c50 = []
        vad_durations = []
        for sample in batch[audio_column_name]:
            res = pipeline({"sample_rate": sample["sampling_rate"],
                            "waveform": torch.tensor(sample["array"][None, :]).to(device).float()})
            
            mask = np.full(res["snr"].shape, False)
            for (segment, _) in res["annotation"].itertracks():
                start = int(segment.start * ratio)
                end = int(segment.end * ratio)
                mask[start:end] = True
            mask =  (~((res["snr"] == 0.0) & (res["c50"] == 0.0)) & mask)

            vad_duration = sum(map(lambda x: x[0].duration, res["annotation"].itertracks()))
            
            if mask.any():
                snr.append(res["snr"][mask].mean())
                c50.append(res["c50"][mask].mean())
            else:
                snr.append(np.full(1, np.nan)[0])
                c50.append(np.full(1, np.nan)[0])

            vad_durations.append(np.float32(vad_duration))
        
        # 16ms window
        batch["snr"] = snr
        batch["c50"] = c50
        batch["speech_duration"] = vad_durations
        
    else:
        res = pipeline({"sample_rate": batch[audio_column_name]["sampling_rate"],
                        "waveform": torch.tensor(batch[audio_column_name]["array"][None, :]).to(device).float()})
        
        mask = np.full(res["snr"].shape, False)
        for (segment, _) in res["annotation"].itertracks():
            start = int(segment.start * ratio)
            end = int(segment.end * ratio)
            mask[start:end] = True
        mask =  (~((res["snr"] == 0.0) & (res["c50"] == 0.0)) & mask)

        vad_duration = sum(map(lambda x: x[0].duration, res["annotation"].itertracks()))     
        
        if mask.any():
            batch["snr"] = res["snr"][mask].mean()
            batch["c50"] = res["c50"][mask].mean()
        else:
            batch["snr"] = np.nan
            batch["c50"] = np.nan
        batch["speech_duration"] = vad_duration
        
    return batch


def clear_snr_cache():
    """Clear cached pipeline and model to free memory."""
    global _model_cache, _pipeline_cache
    _model_cache.clear()
    _pipeline_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()