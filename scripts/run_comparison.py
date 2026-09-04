"""
Run model comparison across multiple ASR models
"""

import sys
import argparse
import yaml
import os
import random
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    print("⚠ python-dotenv not installed. Run: pip install python-dotenv")

# Login to Hugging Face if HF_TOKEN is available (for gated models)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("✓ Logged in to Hugging Face Hub")
    except Exception as e:
        print(f"⚠ HF login failed: {e}")

from src.data import AudioProcessor, GroundTruthLoader
from src.models import WhisperModel, CTCModel, QwenModel, GeminiModel
from src.evaluation import ModelEvaluator
from src.reporting import ReportGenerator
from src.utils.device_utils import print_device_info


def create_model_from_config(config_path):
    """Create model instance from config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    model_type = model_config['type']
    
    if model_type == "whisper":
        return WhisperModel(
            name=model_config['name'],
            model_id=model_config['model_id'],
            language=model_config.get('language', 'hindi')
        )
    elif model_type == "ctc":
        return CTCModel(
            name=model_config['name'],
            model_id=model_config['model_id']
        )
    elif model_type == "qwen":
        return QwenModel(
            name=model_config['name'],
            model_id=model_config['model_id']
        )
    elif model_type == "gemini":
        return GeminiModel(
            name=model_config['name'],
            model_id=model_config['model_id']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple ASR models')
    parser.add_argument('--assets-path', type=str, default='assets',
                       help='Path to assets directory (default: assets)')
    parser.add_argument('--configs', type=str, nargs='+',
                       help='List of config files to evaluate')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['whisper_medium', 'whisper_small', 'ai4bharat', 
                               'indicwav2vec', 'qwen', 'gemini', 'vaani_medium',
                               'oriserve_swift', 'oriserve_apex', 'shunya_hinglish'],
                       help='Predefined models to evaluate')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for reports (default: results)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for processing (default: 1, use 4-8 for GPU)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of audio files to process (default: all)')
    parser.add_argument('--sampling', type=str, default='first',
                       choices=['first', 'random'],
                       help='Sampling method when using --max-files (default: first)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling (default: 42)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ASR MODEL COMPARISON")
    print("=" * 80)
    print_device_info()
    print(f"Assets path: {args.assets_path}")
    print()
    
    # Determine which configs to use
    config_files = []
    
    if args.configs:
        config_files = args.configs
    elif args.models:
        # Map model names to config files
        model_map = {
            'whisper_medium': 'configs/whisper_medium.yaml',
            'whisper_small': 'configs/whisper_small.yaml',
            'ai4bharat': 'configs/ai4bharat_whisper.yaml',
            'indicwav2vec': 'configs/indicwav2vec.yaml',
            'qwen': 'configs/qwen.yaml',
            'gemini': 'configs/gemini.yaml',
            'vaani_medium': 'configs/vaani_medium.yaml',
            'oriserve_swift': 'configs/oriserve_swift.yaml',
            'oriserve_apex': 'configs/oriserve_apex.yaml',
            'shunya_hinglish': 'configs/shunya_hinglish.yaml'
        }
        config_files = [model_map[m] for m in args.models]
    else:
        # Default: compare all available models
        config_files = [
            'configs/whisper_medium.yaml',
            'configs/whisper_small.yaml',
            'configs/ai4bharat_whisper.yaml',
            'configs/vaani_medium.yaml',
            'configs/oriserve_apex.yaml',
            'configs/oriserve_swift.yaml',
            'configs/shunya_hinglish.yaml',
            'configs/indicwav2vec.yaml',
            # Note: Qwen and Gemini require special setup, uncomment if configured:
            # 'configs/qwen.yaml',
            # 'configs/gemini.yaml',
        ]
    
    print(f"📋 Comparing {len(config_files)} models:")
    for cf in config_files:
        print(f"   - {os.path.basename(cf)}")
    print()
    
    # Load ground truth
    gt_loader = GroundTruthLoader(args.assets_path)
    ground_truths = gt_loader.load()
    
    if not ground_truths:
        print("❌ No ground truth data found!")
        return
    
    print(f"✓ Loaded {len(ground_truths)} audio file(s)")
    
    # Apply max-files limit if specified
    if args.max_files and args.max_files < len(ground_truths):
        all_files = list(ground_truths.items())
        
        if args.sampling == 'random':
            random.seed(args.seed)
            selected_files = random.sample(all_files, args.max_files)
            print(f"🎲 Randomly selected {args.max_files} files (seed={args.seed})")
        else:  # first
            selected_files = all_files[:args.max_files]
            print(f"📌 Selected first {args.max_files} files")
        
        ground_truths = dict(selected_files)
        print(f"✓ Processing {len(ground_truths)} audio file(s)")
    
    # Initialize components
    audio_processor = AudioProcessor()
    evaluator = ModelEvaluator(audio_processor, verbose=True, batch_size=args.batch_size)
    
    if args.batch_size > 1:
        print(f"🚀 Batch processing enabled (batch_size={args.batch_size})")
    print()
    
    # Evaluate all models
    all_results = []
    
    for config_file in config_files:
        try:
            print(f"\n{'='*80}")
            print(f"Evaluating: {os.path.basename(config_file)}")
            print(f"{'='*80}")
            
            model = create_model_from_config(config_file)
            result = evaluator.evaluate(model, ground_truths)
            all_results.append(result)
            
        except Exception as e:
            print(f"\n❌ Failed to evaluate {config_file}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append(None)
    
    # Generate reports - detect single vs multi-model mode
    is_single_model = len([r for r in all_results if r is not None]) == 1
    report_gen = ReportGenerator(args.output_dir)
    report_paths = report_gen.generate_all(all_results, single_model=is_single_model)
    report_gen.print_summary(report_paths)
    
    # Print summary based on mode
    if is_single_model:
        # Single model mode - simple completion message
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)
    else:
        # Multi-model comparison mode
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        
        valid_results = [r for r in all_results if r and r['files_processed'] > 0]
        
        if valid_results:
            best_wer = min(valid_results, key=lambda x: x['avg_wer'])
            best_cer = min(valid_results, key=lambda x: x['avg_cer'])
            best_time = min(valid_results, key=lambda x: x['avg_inference_time'])
            
            print(f"\n🏆 BEST MODELS:")
            print(f"   • Best WER: {best_wer['model_name']} ({best_wer['avg_wer']:.4f})")
            print(f"   • Best CER: {best_cer['model_name']} ({best_cer['avg_cer']:.4f})")
            print(f"   • Fastest: {best_time['model_name']} ({best_time['avg_inference_time']:.2f}s)")
        
        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE!")
        print("=" * 80)


if __name__ == "__main__":
    main()
