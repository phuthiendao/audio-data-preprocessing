import os
from typing import List, Dict
from tqdm import tqdm

from prompts.gemini_client import GeminiClient
from prompts.gemini_prompt_template import SYSTEM_PROMPT, build_prompt
from prompts.utils import load_json, load_txt_checkpoint, append_txt, append_jsonl


def generate_descriptions_jsonl(
    records: List[Dict],
    client: GeminiClient,
    output_path: str,
    use_system_prompt: bool = True,
    save_interval: int = 100,
    limit: int = None,
    audio_dir: str = None
) -> List[Dict]:
    # Check for existing progress by reading output file
    processed_filenames = set()
    
    def get_output_filename(fname):
        return os.path.join(audio_dir, fname) if audio_dir else fname

    if os.path.exists(output_path):
        print(f"Reading checkpoint from {output_path}...")
        try:
            existing_data = load_json(output_path)
            processed_filenames = {r.get('filename') for r in existing_data if (r.get('description') or r.get('text'))}
            print(f"Resuming with {len(processed_filenames)} records already processed.")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            
    to_process = [r for r in records if get_output_filename(r.get('filename')) not in processed_filenames]
    
    # Apply limit AFTER checkpoint
    if limit is not None and limit < len(to_process):
        print(f"Limiting to {limit} records (out of {len(to_process)} remaining)")
        to_process = to_process[:limit]
    
    if not to_process:
        print("All records already processed.")
        return []

    print(f"Processing {len(to_process)} remaining records in batches of {save_interval}...")
    
    def chunker(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]
            
    final_results = []
    
    # Process in batches
    for batch_records in chunker(to_process, save_interval):
        batch_prompts = []
        batch_filenames = []
        # Prepare Prompts
        for record in batch_records:
            user_prompt = build_prompt(record)
            if use_system_prompt:
                full_prompt = f"SYSTEM_PROMPT\n\n{user_prompt}".replace("SYSTEM_PROMPT", SYSTEM_PROMPT)
            else:
                full_prompt = user_prompt
            
            batch_prompts.append(full_prompt)
            batch_filenames.append(record.get('filename'))
            
        # Parallel Generation
        batch_descriptions = client.generate_batch(batch_prompts)
        
        # Collect Results
        batch_results = []
        for i, desc in enumerate(batch_descriptions):
            out_filename = get_output_filename(batch_filenames[i])
            result_record = {
                'filename': out_filename,
                'text': desc or "Failed to generate description"
            }
            batch_results.append(result_record)
            final_results.append(result_record)
            
        # Append Batch to File
        from prompts.utils import append_jsonl
        append_jsonl(batch_results, output_path)
            
    return final_results


def generate_descriptions_txt(
        records: List[Dict],
        client: GeminiClient,
        output_path: str,
        use_system_prompt: bool = True,
        limit: int = None,
        audio_dir: str = None
    ) -> int:

    # Load already processed filenames from checkpoint
    processed_filenames = load_txt_checkpoint(output_path) #
    
    
    if processed_filenames:
        print(f"Resuming from {output_path} with {len(processed_filenames)} records already processed.")
    
    # Filter records to process
    def get_output_filename(fname):
        return os.path.join(audio_dir, fname) if audio_dir else fname

    to_process = [r for r in records if get_output_filename(r.get('filename')) not in processed_filenames]
    
    if not to_process:
        print("All records already processed.")
        return len(processed_filenames)
    
    if limit and limit < len(to_process):
        print(f"Limiting to {limit} records (out of {len(to_process)} remaining)")
        to_process = to_process[:limit]
    
    print(f"Processing {len(to_process)} remaining records...")
    
    count = len(processed_filenames)
    
    for record in tqdm(to_process, desc="Generating descriptions"):
        filename = record.get('filename')
        user_prompt = build_prompt(record)
        
        if use_system_prompt:
            full_prompt = f"SYSTEM_PROMPT\n\n{user_prompt}".replace("SYSTEM_PROMPT", SYSTEM_PROMPT)
        else:
            full_prompt = user_prompt
        
        # Generate description
        description = client.generate(full_prompt)
        
        # Construct output parts
        out_filename = get_output_filename(filename)
        
        if description:
            # Format: path/to/file.wav|Description: ...
            out_desc = f"text: {description}"
            append_txt(output_path, out_filename, out_desc)
            count += 1
        else:
            # Handle failure
            out_desc = "Description: Failed to generate description"
            append_txt(output_path, out_filename, out_desc)
            count += 1
    
    print(f"Saved {count} records to {output_path}")
    return count
