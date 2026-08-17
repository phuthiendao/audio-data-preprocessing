import os
import time
import json
import torch
import multiprocessing
import queue
import threading

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from processor.config import SAMPLING_RATE, MIN_AUDIO_DURATION, MAX_AUDIO_DURATION, SNR_MIN_DB
from processor.utils.audio_io import get_audio_files, load_audio_16k, AudioCache, audio_duration_sec
from processor.utils.common import clean_memory, load_transcripts, CheckpointManager, ProgressTracker

from processor.models.batch_manager import BatchAIClassifier
from processor.models.acoustic_features import OpenSmileAcousticFeatures
from processor.core.profile_builder import ProfileBuilder

from dataspeech.gpu_enrichments.pitch import pitch_apply
from dataspeech.gpu_enrichments.snr_and_reverb import snr_apply
from dataspeech.cpu_enrichments.rate import rate_apply


_acoustic_extractor = None


def init_worker():
    """Initialize OpenSmile in each worker process"""
    global _acoustic_extractor
    _acoustic_extractor = OpenSmileAcousticFeatures(sample_rate=SAMPLING_RATE)


def process_audio_file(filepath, text=None):
    """CPU-bound audio loading + OpenSmile"""
    try:
        filename = os.path.basename(filepath)
        audio = load_audio_16k(filepath)

        global _acoustic_extractor
        acoustic = {}
        if _acoustic_extractor:
            acoustic = _acoustic_extractor.compute(filepath, audio_array=audio)

        # Calculate Speaking Rate & Phonemes on CPU
        rate_input = {
            "text": text or "", 
            "audio": {"array": audio, "sampling_rate": SAMPLING_RATE},
            "speech_duration": acoustic.get("speech_duration", 0)
        }
        
        # rate_apply returns the dict with added keys "speaking_rate" and "phonemes"
        rate_res = rate_apply(rate_input, audio_column_name="audio", text_column_name="text")
        
        speaking_rate = rate_res.get("speaking_rate", 0)
        phonemes = rate_res.get("phonemes", "")
        return {
            "filepath": filepath,
            "filename": filename,
            "audio": audio,
            "acoustic": acoustic,
            "speaking_rate": speaking_rate,
            "phonemes": phonemes,
            "text": text,
            "error": None,
        }
    except Exception as e:
        return {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "error": str(e),
        }



def resolve_transcript(filepath, transcripts, is_lazy):
    """Transcript text for filepath, or None if the .txt / entry is missing (no pair)."""
    if is_lazy:
        txt_path = os.path.join(transcripts, os.path.splitext(os.path.basename(filepath))[0] + ".txt")
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                return None
        return None
    return transcripts.get(os.path.basename(filepath))


def sample_skip_reason(filepath, text):
    """Reason to skip a sample, or None if valid: needs a transcript and duration in [MIN, MAX] sec."""
    if not text:
        return "no .txt pair"
    try:
        dur = audio_duration_sec(filepath)
    except Exception as e:
        return f"unreadable audio ({e})"
    if dur < MIN_AUDIO_DURATION:
        return f"too short ({dur:.2f}s)"
    if dur > MAX_AUDIO_DURATION:
        return f"too long ({dur:.2f}s)"
    return None


# VoiceProcessor
class VoiceProcessor:
    def __init__(self, output_file, workers: int = 20):
        self.output_file = output_file
        self.workers = workers

        self.ai_classifier = BatchAIClassifier()
        self.profile_builder = ProfileBuilder()
        self.audio_cache = AudioCache()
        self.ckpt_manager = CheckpointManager(output_file)

        self.process_executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            mp_context=multiprocessing.get_context("spawn"),
        )

    def __del__(self):
        if hasattr(self, "process_executor"):
            self.process_executor.shutdown(wait=False)

    # GPU BATCH PROCESSING
    def _gpu_process_batch(self, cpu_results):
        if not cpu_results:
            return []

        audio_batch_list = []
        filepaths = []
        
        valid_cpu_results = [r for r in cpu_results if not r.get("error")]

        for item in valid_cpu_results:
            audio_batch_list.append(
                {"array": item["audio"], "sampling_rate": SAMPLING_RATE}
            )
            filepaths.append(item["filepath"])
            self.audio_cache.preload(item["filepath"], item["audio"], SAMPLING_RATE)

        if not valid_cpu_results:
            return []

        ds_batch = {
            "audio": audio_batch_list,
            "text": [item["text"] or "" for item in valid_cpu_results],
        }

        # GPU processing with retry on smaller batches
        gpu_success = False
        rank = 0 if torch.cuda.is_available() else None
        try:
            with torch.no_grad():
                ds_batch = pitch_apply(ds_batch, rank=rank)
                ds_batch = snr_apply(ds_batch, rank=rank)
            gpu_success = True
        except Exception as e:
            print(f"Batch failed: {e}")
            # Try processing one-by-one as fallback
            if len(valid_cpu_results) > 1:
                print(f"Processing {len(valid_cpu_results)} items individually")
                all_profiles = []
                for item in valid_cpu_results:
                    single_result = self._gpu_process_batch([item])
                    all_profiles.extend(single_result)
                return all_profiles
            else:
                print(f"Skipping file: {valid_cpu_results[0]['filename']}")
                return []

        if not gpu_success:
            return []

        # AI predictions with error handling
        try:
            ai_predictions = self.ai_classifier.predict_batch_from_cache(
                filepaths, self.audio_cache
            )
        except Exception as e:
            print(f"Batch prediction failed: {e}")
            # Fallback to individual predictions
            ai_predictions = []
            for fp in filepaths:
                try:
                    pred = self.ai_classifier.predict_batch_from_cache([fp], self.audio_cache)
                    ai_predictions.extend(pred)
                except:
                    ai_predictions.append({"gender": "unknown", "accent": "unknown", "emotion": "unknown", "filepath": fp})

        final_profiles = []

        for i, item in enumerate(valid_cpu_results):
            # Safe access to batched results
            try:
                # Safely get AI prediction with fallback
                ai_pred = ai_predictions[i] if i < len(ai_predictions) else {
                    "gender": "unknown", "accent": "unknown", "emotion": "unknown"
                }
                
                item["pitch_mean"] = float(ds_batch["utterance_pitch_mean"][i])
                item["pitch_std"] = float(ds_batch["utterance_pitch_std"][i])
                item["pitch_std_st"] = float(ds_batch["utterance_pitch_std_st"][i])
                item["pitch_range_st"] = float(ds_batch["utterance_pitch_range_st"][i])
                item["snr"] = float(ds_batch["snr"][i])
                item["c50"] = float(ds_batch["c50"][i])
                item["speech_duration"] = float(ds_batch["speech_duration"][i])

                # Stage 2: drop low-SNR samples (post-extraction acoustic filter)
                if item["snr"] < SNR_MIN_DB:
                    print(f"[drop] {item['filename']}: SNR {item['snr']:.1f} dB < {SNR_MIN_DB} dB")
                    continue

                profile = self.profile_builder.build(item, ai_pred)
                final_profiles.append(profile)
            except Exception as e:
                print(f"Error building profile for {item['filename']}: {e}")

        return final_profiles

    # SEQUENTIAL PROCESSING (CPU-friendly: one file at a time, no pool/threads)
    def process_all_sequential(self, audio_dir, transcript_file=None):
        init_worker()  # init OpenSMILE in THIS process (ProcessPoolExecutor unused here)

        transcripts = load_transcripts(transcript_file)
        is_lazy_transcript = isinstance(transcripts, str)

        _, processed = self.ckpt_manager.load_checkpoint()
        tracker = ProgressTracker(total=0)
        num_prior_processed = len(processed)

        clean_memory(force=True)

        skipped = 0
        for filepath in get_audio_files(audio_dir):
            fname = os.path.basename(filepath)
            if fname in processed:
                continue

            text_content = resolve_transcript(filepath, transcripts, is_lazy_transcript)
            reason = sample_skip_reason(filepath, text_content)
            if reason:
                skipped += 1
                print(f"[skip] {fname}: {reason}")
                continue

            t0 = time.time()
            cpu_result = process_audio_file(filepath, text_content)
            if cpu_result.get("error"):
                tracker.update(success=False)
                print(f"Failed: {fname} | {cpu_result['error']}")
                continue

            profiles = self._gpu_process_batch([cpu_result])
            if profiles:
                self.ckpt_manager.save_checkpoint(profiles)
                tracker.update(success=True)
            else:
                tracker.update(success=False)

            self.audio_cache.clear()

            current = num_prior_processed + tracker.processed
            print(f"Processed: {current} | Failed: {tracker.failed} | {fname} | {time.time()-t0:.2f}s")

        if skipped:
            print(f"Skipped {skipped} file(s): no .txt pair or duration outside [{MIN_AUDIO_DURATION}s, {MAX_AUDIO_DURATION}s]")
        clean_memory(force=True)

    # PIPELINED PROCESSING
    def process_all(self, audio_dir, batch_size=64, transcript_file=None):

        files_gen = get_audio_files(audio_dir)
        transcripts = load_transcripts(transcript_file)
        is_lazy_transcript = isinstance(transcripts, str)

        _, processed = self.ckpt_manager.load_checkpoint()
        tracker = ProgressTracker(total=0)
        
        # Re-introducing background file counting for accurate
        num_prior_processed = len(processed)
        total_files_est = 0
        total_counted = False
        
        def count_files_bg():
            nonlocal total_files_est, total_counted
            count = 0
            for _ in get_audio_files(audio_dir):
                count += 1
                if count % 10000 == 0:
                    total_files_est = count
            total_files_est = count
            total_counted = True
            
        bg_counter = threading.Thread(target=count_files_bg, daemon=True)
        bg_counter.start()

        # OPTIMIZATION: Increased Queue Size for Buffering
        ready_queue = queue.Queue(maxsize=256) 
        STOP = object()

        # GPU WORKER THREAD
        def gpu_worker():
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
            torch.set_grad_enabled(False)
            
            # Start clean
            clean_memory(force=True)

            batch = []
            
            def run_gpu_batch(batch_list):
                 # Separate valid and error items
                 valid_items = [x for x in batch_list if not x.get("error")]
                 error_items = [x for x in batch_list if x.get("error")]
                 
                 # Record errors immediately
                 for _ in error_items:
                     tracker.update(success=False)

                 t0 = time.time()
                 if valid_items:
                    profiles = self._gpu_process_batch(valid_items)
                    if profiles:
                        self.ckpt_manager.save_checkpoint(profiles)
                        for _ in profiles:
                            tracker.update(success=True)
                 
                 duration = time.time() - t0
                 
                 current_total_processed = num_prior_processed + tracker.processed
                 total_files_str = str(total_files_est) if total_counted else f">{total_files_est}"
                 
                 print(f"Processed: {current_total_processed}/{total_files_str} | Failed: {tracker.failed} | BatchTime: {duration:.2f}s")

            while True:
                item = ready_queue.get()
                if item is STOP:
                    break

                batch.append(item)

                if len(batch) >= batch_size:
                    run_gpu_batch(batch)

                    batch.clear()
                    self.audio_cache.clear()

            if batch:
                run_gpu_batch(batch)
                
            clean_memory(force=True)

        gpu_executor = ThreadPoolExecutor(max_workers=1)
        gpu_executor.submit(gpu_worker)


        # CPU SUBMISSION LOOP
        futures = set()
        prefetch = self.workers * 2 

        def submit_next():
            try:
                f_path = next(files_gen)
            except StopIteration:
                return False

            fname = os.path.basename(f_path)
            if fname in processed:
                return True

            text_content = resolve_transcript(f_path, transcripts, is_lazy_transcript)
            reason = sample_skip_reason(f_path, text_content)
            if reason:
                print(f"[skip] {fname}: {reason}")
                return True

            fut = self.process_executor.submit(
                process_audio_file, f_path, text_content
            )
            futures.add(fut)
            return True

        # Initial seed
        while len(futures) < prefetch:
             if not submit_next():
                 break

        # MAIN COLLECTION LOOP
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            
            for fut in done:
                futures.remove(fut)
                try:
                    res = fut.result()
                    if res and not res.get("error"):
                        ready_queue.put(res)
                    elif res and res.get("error"):
                         ready_queue.put({"error": res['error'], "filepath": res['filepath'], "filename": res['filename']})

                except Exception as e:
                    print(f"{e}")

                submit_next()

        # shutdown
        ready_queue.put(STOP)
        gpu_executor.shutdown(wait=True)

