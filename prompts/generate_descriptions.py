#!/usr/bin/env python3

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.gemini_client import GeminiClient
from prompts.utils import load_json
from prompts.processors import generate_descriptions_txt, generate_descriptions_jsonl


def main():
    parser = argparse.ArgumentParser(description="Generate natural language descriptions using Gemini")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file with audio metadata")
    parser.add_argument("--output", required=True, help="Output file (.jsonl or .txt)")
    parser.add_argument("--api-key", help="Google Gemini API key")
    parser.add_argument(
        "--model",
        default="models/gemini-2.5-flash",
        choices=["models/gemini-2.5-flash", "models/gemini-flash-latest", "models/gemini-2.0-flash-lite", "models/gemini-2.0-flash"],
        help="Gemini model to use (default: models/gemini-2.5-flash)"
    )
    parser.add_argument("--limit", type=int, help="Limit number of records to process (for testing)")
    parser.add_argument("--no-system-prompt", action="store_true", help="Don't include system prompt (not recommended)")
    parser.add_argument("--audio-dir", help="Directory path to prepend to audio filenames in output")
    
    args = parser.parse_args()
    
    # Determine output format
    output_format = "jsonl"
    if args.output.endswith('.txt'):
        output_format = "txt"
    elif args.output.endswith('.jsonl'):
        output_format = "jsonl"
    else:
        print("Error: Output file must be .jsonl or .txt")
        sys.exit(1)
    
    # Load input data
    try:
        records = load_json(args.input)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    if args.limit and output_format == "jsonl":
        print(f"Limit of {args.limit} will be applied after checkpoint loading.")
    
    print(f"Total records in input: {len(records)}")
    print(f"Output format: {output_format.upper()}")
    
    # Initialize Gemini client
    print(f"Initializing Gemini ({args.model})...")
    try:
        client = GeminiClient(api_key=args.api_key, model=args.model)
    except ValueError as e:
         print(f"Error: {e}")
         print("SET GOOGLE_API_KEY environment")
         sys.exit(1)
    

    try:
        if output_format == "txt":
            generate_descriptions_txt(
                records,
                client,
                output_path=args.output,
                use_system_prompt=not args.no_system_prompt,
                limit=args.limit,
                audio_dir=args.audio_dir
            )
        else:
             generate_descriptions_jsonl(
                records,
                client,
                output_path=args.output,
                use_system_prompt=not args.no_system_prompt,
                save_interval=100,
                limit=args.limit,
                audio_dir=args.audio_dir
            )
    except KeyboardInterrupt:
        print("\nProgress has been saved to checkpoint.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError occurred: {e}")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)
    
    # Results
    print(f"\nCompleted! Saved to {args.output}")

if __name__ == "__main__":
    main()
