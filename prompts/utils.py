import json
import os
from typing import List, Dict

def load_json(filepath: str) -> List[Dict]:
    """Load JSON or JSONL file."""
    if filepath.endswith('.jsonl'):
        return load_jsonl(filepath)
        
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def append_jsonl(data: List[Dict], filepath: str):
    """Append list of dicts to JSONL file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_txt_checkpoint(filepath: str) -> set:
    """Load processed filenames from TXT checkpoint."""
    processed = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    filename = line.split('|', 1)[0]
                    processed.add(filename)
    return processed


def append_txt(filepath: str, filename: str, description: str):
    clean_desc = ' '.join(description.replace('\n', ' ').split())
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"{filename}|{clean_desc}\n")
