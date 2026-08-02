import torch 
import penn


# Here we'll use a 10 millisecond hopsize
hopsize = .01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1000.

# Select a checkpoint to use for inference. Selecting None will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = None

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
center = 'half-hop'

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = .065


def _semitone_stats(pitch):
    # F0 (Hz) contour -> variation (std) & p80-p20 range in semitones (perceptually uniform)
    st = 12.0 * torch.log2(pitch.flatten() / 27.5)
    return st.std(), torch.quantile(st, 0.8) - torch.quantile(st, 0.2)


def pitch_apply(batch, rank=None, audio_column_name="audio", output_column_name="utterance_pitch", penn_batch_size=4096):
    if isinstance(batch[audio_column_name], list):  
        utterance_pitch_mean = []
        utterance_pitch_std = []
        utterance_pitch_std_st = []
        utterance_pitch_range_st = []
        for sample in batch[audio_column_name]:
            # Infer pitch and periodicity
            pitch, periodicity = penn.from_audio(
                torch.tensor(sample["array"][None, :]).float(),
                sample["sampling_rate"],
                hopsize=hopsize,
                fmin=fmin,
                fmax=fmax,
                checkpoint=checkpoint,
                batch_size=penn_batch_size,
                center=center,
                interp_unvoiced_at=interp_unvoiced_at,
                gpu=(rank or 0)% torch.cuda.device_count() if torch.cuda.device_count() > 0 else rank
                )
            
            utterance_pitch_mean.append(pitch.mean().cpu())
            utterance_pitch_std.append(pitch.std().cpu())
            std_st, range_st = _semitone_stats(pitch)
            utterance_pitch_std_st.append(std_st.cpu())
            utterance_pitch_range_st.append(range_st.cpu())

        batch[f"{output_column_name}_mean"] = utterance_pitch_mean
        batch[f"{output_column_name}_std"] = utterance_pitch_std
        batch[f"{output_column_name}_std_st"] = utterance_pitch_std_st
        batch[f"{output_column_name}_range_st"] = utterance_pitch_range_st
    else:
        sample = batch[audio_column_name]
        pitch, periodicity = penn.from_audio(
                torch.tensor(sample["array"][None, :]).float(),
                sample["sampling_rate"],
                hopsize=hopsize,
                fmin=fmin,
                fmax=fmax,
                checkpoint=checkpoint,
                batch_size=penn_batch_size,
                center=center,
                interp_unvoiced_at=interp_unvoiced_at,
                gpu=(rank or 0)% torch.cuda.device_count() if torch.cuda.device_count() > 0 else rank
                )        
        batch[f"{output_column_name}_mean"] = pitch.mean().cpu()
        batch[f"{output_column_name}_std"] = pitch.std().cpu()
        std_st, range_st = _semitone_stats(pitch)
        batch[f"{output_column_name}_std_st"] = std_st.cpu()
        batch[f"{output_column_name}_range_st"] = range_st.cpu()

    return batch
