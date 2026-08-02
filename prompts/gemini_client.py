import os
import time
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types


import threading

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-lite"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")
        
        self.client = genai.Client(api_key=self.api_key)
        
        if model.startswith("models/"):
            self.model_name = model.replace("models/", "")
        else:
            self.model_name = model
        
        # Paid Tier limits 
        if "2.0-flash-lite" in model.lower():
            self.requests_per_minute = 6000
        elif "flash" in model.lower():
            self.requests_per_minute = 6000
        elif "pro" in model.lower():
            self.requests_per_minute = 360
        else:
            self.requests_per_minute = 60
            
        print(f"Configured GeminiClient: model={self.model_name}, limits={self.requests_per_minute} RPM")
        
        self.last_request_time = 0
        self.min_request_interval = 60.0 / self.requests_per_minute
        self._lock = threading.Lock()
    
    def _wait_for_rate_limit(self):
        with self._lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()
    

    def generate(self, prompt: str, max_retries: int = 5) -> Optional[str]:
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=200,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    )
                )
                
                if response.text:
                    return response.text.strip()
                else:
                    return None
                    
            except Exception as e:
                error_str = str(e)
                # print(f"Attempt {attempt + 1}/{max_retries} failed: {error_str[:100]}")
                
                if "429" in error_str:
                    wait_time = 5 + (2 ** attempt) # Reduced initial wait time
                    print(f"Rate limited (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Error: {error_str[:50]}... Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts: {error_str[:50]}")
                    return None
        
        return None
    
    def generate_batch(self, prompts: List[str], max_retries: int = 3, max_workers: int = 150) -> List[Optional[str]]:
        """Generate multiple descriptions in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        results = [None] * len(prompts)
        
        # Helper to preserve order
        def process_one(idx, p):
            return idx, self.generate(p, max_retries)

        print(f"Sending requests with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all
            future_to_idx = {executor.submit(process_one, i, p): i for i, p in enumerate(prompts)}
            
            # Progress bar
            for future in tqdm(as_completed(future_to_idx), total=len(prompts), desc="Gemini Parallel"):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"Worker exception: {e}")
                    
        return results
