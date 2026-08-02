import argparse
import sys
import os
import time

sys.path.append(os.getcwd())

# Apply PyTorch compatibility patch before importing models
from unified_analysis.utils.torch_compat import ensure_compatibility
ensure_compatibility()

from unified_analysis.core.processor import VoiceProcessor
from unified_analysis.scripts.convert_to_text import ProfileToTextMapper

def cmd_process(args):
    #print(f"Starting Processing: {args.input}")
    processor = VoiceProcessor(output_file=args.output, workers=args.workers)
    
    transcript = args.transcript if hasattr(args, 'transcript') else None
    
    start_time = time.time()
    if getattr(args, "sequential", False):
        processor.process_all_sequential(
            audio_dir=args.input,
            transcript_file=transcript
        )
    else:
        processor.process_all(
            audio_dir=args.input,
            batch_size=args.batch_size,
            transcript_file=transcript
        )
    print(f"Total Time: {round(time.time() - start_time, 2)}s")

def cmd_convert(args):
    #print(f"Starting Conversion: {args.input} -> {args.output}")
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
        
    mapper = ProfileToTextMapper()
    mapper.process(args.input, args.output)

def main():
    
    parser = argparse.ArgumentParser(description="Unified Audio Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process Command
    proc_parser = subparsers.add_parser("process", help="Analyze audio files and generate profiles")
    proc_parser.add_argument("--input", required=True, help="Input directory containing audio files")
    proc_parser.add_argument("--output", required=True, help="Output JSONL file path")
    proc_parser.add_argument("--transcript", default=None, help="Optional transcript JSON file")
    proc_parser.add_argument("--workers", type=int, default=16, help="Number of CPU workers")
    proc_parser.add_argument("--batch_size", type=int, default=128, help="Batch size for GPU inference")
    proc_parser.add_argument("--sequential", action="store_true", help="Process one file at a time, no pool/threads (recommended for CPU / few cores)")
    
    # CLI Convert   
    conv_parser = subparsers.add_parser("convert", help="Convert JSON profiles to text descriptions")
    conv_parser.add_argument("--input", required=True, help="Input profiles JSON file")
    conv_parser.add_argument("--output", default=None, help="Output text JSON file (optional)")
    
    args = parser.parse_args()
    
    if args.command == "process":
        cmd_process(args)
    elif args.command == "convert":
        cmd_convert(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
