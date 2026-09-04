#!/usr/bin/env python3
"""
Create Devanagari ground truth from Roman script ground truth.

This script uses OpenAI API to convert Roman script Hindi transcriptions
to proper Devanagari script for fair comparison with models that output Devanagari.

Usage:
    python scripts/create_devanagari_ground_truth.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def convert_to_devanagari(roman_text: str, client: OpenAI) -> str:
    """
    Convert Roman script Hindi to Devanagari using OpenAI API.
    
    Args:
        roman_text: Hindi text in Roman script
        client: OpenAI client instance
        
    Returns:
        str: Text converted to Devanagari script
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use mini for cost efficiency
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert Hindi transliterator. Your task is to convert Hinglish (Hindi written in Roman/Latin script) to proper Devanagari script.

Rules:
1. Convert Hindi words to Devanagari
2. Keep English words (like Doctor, Address, etc.) in Devanagari phonetic form
3. Keep numbers as numerals (31, 65, etc.)
4. Preserve punctuation and structure
5. Handle medical terms appropriately
6. Output ONLY the Devanagari text, nothing else."""
                },
                {
                    "role": "user",
                    "content": f"Convert this Hinglish text to Devanagari:\n\n{roman_text}"
                }
            ],
            temperature=0,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"  ✗ Error converting: {e}")
        return ""


def process_ground_truth_file(input_file: str, output_file: str):
    """
    Process ground truth file and create dual-script version.
    
    Args:
        input_file: Path to input ground truth file (Roman script)
        output_file: Path to output dual-script ground truth file
    """
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        print("   Please add your OpenAI API key to continue.")
        return
    
    client = OpenAI(api_key=api_key)
    
    print("=" * 80)
    print("DEVANAGARI GROUND TRUTH GENERATOR")
    print("=" * 80)
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}\n")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    processed_lines = []
    total_lines = len([l for l in lines if l.strip() and '|' in l])
    current = 0
    
    print(f"Processing {total_lines} transcriptions...\n")
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            processed_lines.append("")
            continue
        
        # Parse line
        if "|" not in line:
            processed_lines.append(line)
            continue
        
        current += 1
        
        # Split into parts
        parts = line.split("|")
        if len(parts) < 2:
            processed_lines.append(line)
            continue
        
        audio_file = parts[0].strip()
        roman_text = parts[1].strip()
        
        # Get diagnosis if present
        diagnosis = parts[2].strip() if len(parts) > 2 else ""
        
        print(f"[{current}/{total_lines}] {audio_file}")
        print(f"  Roman: {roman_text[:60]}...")
        
        # Convert to Devanagari
        devanagari_text = convert_to_devanagari(roman_text, client)
        
        if devanagari_text:
            print(f"  देवनागरी: {devanagari_text[:60]}...")
            
            # Format: audio_file | roman_text | devanagari_text | diagnosis
            if diagnosis:
                new_line = f"{audio_file} | {roman_text} | {devanagari_text} | {diagnosis}"
            else:
                new_line = f"{audio_file} | {roman_text} | {devanagari_text}"
            
            processed_lines.append(new_line)
            print("  ✓ Converted")
        else:
            # Keep original if conversion failed
            processed_lines.append(line)
            print("  ⚠ Conversion failed, keeping original")
        
        print()
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines))
    
    print("=" * 80)
    print(f"✓ Dual-script ground truth created: {output_file}")
    print(f"  Processed: {current} transcriptions")
    print("=" * 80)


def main():
    """Main function"""
    # File paths
    input_file = "assets/grounds_truth.txt"
    output_file = "assets/grounds_truth_dual.txt"
    
    # Confirm with user
    print("\n⚠️  This will use OpenAI API and may incur costs.")
    with open(input_file, 'r', encoding='utf-8') as f:
        line_count = len(f.readlines())
    print(f"   Estimated cost: ~$0.01-0.05 for {line_count} lines")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    process_ground_truth_file(input_file, output_file)
    
    print("\n📝 Next steps:")
    print("   1. Review the generated file: assets/grounds_truth_dual.txt")
    print("   2. Manually verify a few Devanagari conversions")
    print("   3. Update ground_truth_loader.py to use dual-script format")


if __name__ == "__main__":
    main()
