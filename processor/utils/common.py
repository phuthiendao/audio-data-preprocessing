import os
import gc
import json
import time
from datetime import datetime
import torch

def clean_memory(force=False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if force:
            torch.cuda.synchronize()

def load_transcripts(transcript_path: str | None) -> dict[str, str]:
    if not transcript_path:
        return {}
    
    # Case 1: JSON file
    if os.path.isfile(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Case 2: Directory txt files (optimized for streaming)
    if os.path.isdir(transcript_path):
        print(f"Transcript is directory: {transcript_path}. Using lazy loading.")
        return transcript_path
    
    raise FileNotFoundError(f"Transcript path not found: {transcript_path}")

class CheckpointManager:
    def __init__(self, output_path):
        self.output_path = output_path
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
    def load_checkpoint(self):
        processed_files = set()
        
        if os.path.exists(self.output_path):
            print(f"Loading checkpoint from {self.output_path}...")
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            item = json.loads(line)
                            if "filename" in item:
                                processed_files.add(item["filename"])
                        except json.JSONDecodeError:
                            continue # Skip bad lines
                
                print(f"Resuming: {len(processed_files)} files already processed.")
            except Exception as e:
                print(f"Warning: Failed to read existing file: {e}")
        
        return [], processed_files 
    
    def save_checkpoint(self, new_results, processed_files=None):
        if not new_results:
            return

        try:
            with open(self.output_path, "a", encoding="utf-8") as f:
                for item in new_results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Failed to append results: {e}")
    
class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.processed = 0
        self.failed = 0
        self.start_time = time.time()
        
    def update(self, success=True):
        self.processed += 1
        if not success:
            self.failed += 1
    
    def set_total(self, total):
        self.total = total
        
    def get_summary(self):
        elapsed = time.time() - self.start_time
        total_str = f"/{self.total}" if self.total > 0 else ""
        return {
            "processed": self.processed,
            "total_str": total_str,
            "failed": self.failed,
            "elapsed": round(elapsed, 1),
        }
