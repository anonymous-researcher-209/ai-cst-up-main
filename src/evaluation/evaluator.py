"""Model evaluation orchestration"""

import time
import numpy as np
from .metrics import calculate_wer, calculate_cer

# Import transliteration utilities
try:
    from ..utils.transliteration import (
        normalize_for_comparison,
        detect_script,
        devanagari_to_roman
    )
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False


class ModelEvaluator:
    """Evaluate ASR models on audio files"""
    
    def __init__(self, audio_processor, verbose=True, batch_size=1):
        """
        Initialize ModelEvaluator
        
        Args:
            audio_processor: AudioProcessor instance
            verbose: Print progress messages
            batch_size: Number of files to process in a batch (1=sequential)
        """
        self.audio_processor = audio_processor
        self.verbose = verbose
        self.batch_size = batch_size
    
    def evaluate(self, model, ground_truths):
        """
        Evaluate a model on all ground truth data
        
        Args:
            model: ASR model instance (must have transcribe method)
            ground_truths: Dict mapping audio paths to transcriptions
            
        Returns:
            dict: Evaluation results with metrics
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Evaluating: {model.name}")
            print(f"{'='*80}")
        
        # Load model
        load_start = time.time()
        model_info = model.load()
        load_time = time.time() - load_start
        
        if self.verbose:
            print(f"✓ Model loaded in {load_time:.2f}s")
            if model_info:
                if 'parameters_M' in model_info:
                    print(f"✓ Parameters: {model_info['parameters_M']:.2f}M")
                if 'device' in model_info:
                    print(f"✓ Device: {model_info['device']}")
        
        # Evaluate on all audio files
        results = {
            'model_name': model.name,
            'model_id': model.model_id,
            'model_type': model.model_type,
            'parameters_M': model_info.get('parameters_M', 0) if model_info else 0,
            'load_time': load_time,
            'predictions': {},
            'wer_scores': [],
            'cer_scores': [],
            'inference_times': [],
            'total_time': 0
        }
        
        if self.verbose:
            print(f"\nProcessing {len(ground_truths)} audio file(s)...")
        
        total_start = time.time()
        
        # Check if model supports batch processing
        use_batching = (self.batch_size > 1 and 
                       hasattr(model, 'transcribe_batch') and 
                       callable(getattr(model, 'transcribe_batch')))
        
        if use_batching:
            results = self._evaluate_batched(model, ground_truths, results, total_start)
        else:
            results = self._evaluate_sequential(model, ground_truths, results, total_start)
        
        # Cleanup model
        model.cleanup()
        
        return results
    
    def _evaluate_sequential(self, model, ground_truths, results, total_start):
        """Evaluate files one by one (original method)"""
        for audio_path, gt_text_or_dict in ground_truths.items():
            try:
                # Process audio
                audio = self.audio_processor.load_and_process(audio_path)
                if audio is None:
                    if self.verbose:
                        print(f"  ✗ Failed to load: {audio_path}")
                    continue
                
                # Transcribe
                inference_start = time.time()
                predicted = model.transcribe(audio, audio_path)
                inference_time = time.time() - inference_start
                
                if not predicted:
                    if self.verbose:
                        print(f"  ⚠ Empty transcription for: {audio_path}")
                    continue
                
                # Detect predicted script and select matching ground truth
                if TRANSLITERATION_AVAILABLE:
                    pred_script = detect_script(predicted)
                else:
                    pred_script = 'unknown'
                
                # Select appropriate ground truth based on predicted script
                if isinstance(gt_text_or_dict, dict):
                    # Dual-script ground truth - select matching version
                    if pred_script == 'devanagari' and 'devanagari' in gt_text_or_dict:
                        gt_text = gt_text_or_dict['devanagari']
                        gt_script = 'devanagari'
                    elif pred_script == 'roman' and 'roman' in gt_text_or_dict:
                        gt_text = gt_text_or_dict['roman']
                        gt_script = 'roman'
                    else:
                        # Fallback to roman if script unknown or not available
                        gt_text = gt_text_or_dict.get('roman', gt_text_or_dict.get('devanagari', ''))
                        gt_script = detect_script(gt_text) if TRANSLITERATION_AVAILABLE else 'unknown'
                else:
                    # Single-script ground truth
                    gt_text = gt_text_or_dict
                    gt_script = detect_script(gt_text) if TRANSLITERATION_AVAILABLE else 'unknown'
                
                # Calculate metrics
                wer_score = calculate_wer(gt_text, predicted)
                cer_score = calculate_cer(gt_text, predicted)
                
                # Get normalized (transliterated) versions
                if TRANSLITERATION_AVAILABLE:
                    gt_normalized, pred_normalized = normalize_for_comparison(gt_text, predicted)
                    # If prediction is in Devanagari, also store its romanized form
                    pred_roman = devanagari_to_roman(predicted) if pred_script == 'devanagari' else predicted
                else:
                    gt_normalized = gt_text
                    pred_normalized = predicted
                    pred_roman = predicted
                
                # Store results
                results['predictions'][audio_path] = {
                    'predicted': predicted,
                    'actual': gt_text,
                    'predicted_roman': pred_roman,  # Transliterated version
                    'predicted_script': pred_script,
                    'actual_script': gt_script,
                    'normalized_predicted': pred_normalized,
                    'normalized_actual': gt_normalized,
                    'wer': wer_score,
                    'cer': cer_score,
                    'inference_time': inference_time
                }
                
                results['wer_scores'].append(wer_score)
                results['cer_scores'].append(cer_score)
                results['inference_times'].append(inference_time)
                
                if self.verbose:
                    filename = audio_path.split('\\')[-1]  # Get filename only
                    print(f"  ✓ {filename}: WER={wer_score:.3f}, CER={cer_score:.3f}, Time={inference_time:.2f}s")
                
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Error processing {audio_path}: {e}")
        
        results['total_time'] = time.time() - total_start
        
        # Calculate averages
        if results['wer_scores']:
            results['avg_wer'] = np.mean(results['wer_scores'])
            results['avg_cer'] = np.mean(results['cer_scores'])
            results['avg_inference_time'] = np.mean(results['inference_times'])
            results['files_processed'] = len(results['wer_scores'])
        else:
            results['avg_wer'] = None
            results['avg_cer'] = None
            results['avg_inference_time'] = None
            results['files_processed'] = 0
        
        if self.verbose:
            print(f"\n✓ Evaluation completed in {results['total_time']:.2f}s")
            if results['files_processed'] > 0:
                print(f"✓ Average WER: {results['avg_wer']:.4f}")
                print(f"✓ Average CER: {results['avg_cer']:.4f}")
        
        return results
    
    def _evaluate_batched(self, model, ground_truths, results, total_start):
        """Evaluate files in batches for speedup"""
        audio_paths = list(ground_truths.keys())
        gt_texts = list(ground_truths.values())
        
        num_batches = (len(audio_paths) + self.batch_size - 1) // self.batch_size
        
        if self.verbose:
            print(f"Using batch processing (batch_size={self.batch_size}, {num_batches} batches)")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(audio_paths))
            
            batch_paths = audio_paths[start_idx:end_idx]
            batch_gts = gt_texts[start_idx:end_idx]
            
            try:
                # Load batch
                waveforms, valid_paths, failed_paths = self.audio_processor.load_batch(batch_paths)
                
                if not waveforms:
                    continue
                
                # Pad batch
                padded_batch = self.audio_processor.pad_batch(waveforms)
                
                # Transcribe batch
                inference_start = time.time()
                predictions = model.transcribe_batch(padded_batch)
                inference_time = time.time() - inference_start
                
                # Process results
                for path, pred, gt_text_or_dict in zip(valid_paths, predictions, 
                                         [batch_gts[batch_paths.index(p)] for p in valid_paths]):
                    if not pred:
                        if self.verbose:
                            print(f"  ⚠ Empty transcription for: {path}")
                        continue
                    
                    # Detect predicted script and select matching ground truth
                    if TRANSLITERATION_AVAILABLE:
                        pred_script = detect_script(pred)
                    else:
                        pred_script = 'unknown'
                    
                    # Select appropriate ground truth based on predicted script
                    if isinstance(gt_text_or_dict, dict):
                        # Dual-script ground truth - select matching version
                        if pred_script == 'devanagari' and 'devanagari' in gt_text_or_dict:
                            gt_text = gt_text_or_dict['devanagari']
                            gt_script = 'devanagari'
                        elif pred_script == 'roman' and 'roman' in gt_text_or_dict:
                            gt_text = gt_text_or_dict['roman']
                            gt_script = 'roman'
                        else:
                            # Fallback to roman if script unknown or not available
                            gt_text = gt_text_or_dict.get('roman', gt_text_or_dict.get('devanagari', ''))
                            gt_script = detect_script(gt_text) if TRANSLITERATION_AVAILABLE else 'unknown'
                    else:
                        # Single-script ground truth
                        gt_text = gt_text_or_dict
                        gt_script = detect_script(gt_text) if TRANSLITERATION_AVAILABLE else 'unknown'
                    
                    # Calculate metrics
                    wer_score = calculate_wer(gt_text, pred)
                    cer_score = calculate_cer(gt_text, pred)
                    
                    # Get normalized (transliterated) versions
                    if TRANSLITERATION_AVAILABLE:
                        gt_normalized, pred_normalized = normalize_for_comparison(gt_text, pred)
                        # If prediction is in Devanagari, also store its romanized form
                        pred_roman = devanagari_to_roman(pred) if pred_script == 'devanagari' else pred
                    else:
                        gt_normalized = gt_text
                        pred_normalized = pred
                        pred_roman = pred
                    
                    # Store results
                    results['predictions'][path] = {
                        'predicted': pred,
                        'actual': gt_text,
                        'predicted_roman': pred_roman,
                        'predicted_script': pred_script,
                        'actual_script': gt_script,
                        'normalized_predicted': pred_normalized,
                        'normalized_actual': gt_normalized,
                        'wer': wer_score,
                        'cer': cer_score,
                        'inference_time': inference_time / len(predictions)
                    }
                    
                    results['wer_scores'].append(wer_score)
                    results['cer_scores'].append(cer_score)
                    results['inference_times'].append(inference_time / len(predictions))
                    
                    if self.verbose:
                        filename = path.split('\\')[-1]
                        print(f"  ✓ {filename}: WER={wer_score:.3f}, CER={cer_score:.3f}")
                
                # Report failed files
                for path in failed_paths:
                    if self.verbose:
                        print(f"  ✗ Failed to load: {path}")
                
                if self.verbose and batch_idx % 5 == 0:
                    print(f"  Progress: {end_idx}/{len(audio_paths)} files processed")
                    
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Error processing batch {batch_idx}: {e}")
        
        results['total_time'] = time.time() - total_start
        
        # Calculate averages
        if results['wer_scores']:
            results['avg_wer'] = np.mean(results['wer_scores'])
            results['avg_cer'] = np.mean(results['cer_scores'])
            results['avg_inference_time'] = np.mean(results['inference_times'])
            results['files_processed'] = len(results['wer_scores'])
        else:
            results['avg_wer'] = None
            results['avg_cer'] = None
            results['avg_inference_time'] = None
            results['files_processed'] = 0
        
        if self.verbose:
            print(f"\n✓ Evaluation completed in {results['total_time']:.2f}s")
            if results['files_processed'] > 0:
                print(f"✓ Average WER: {results['avg_wer']:.4f}")
                print(f"✓ Average CER: {results['avg_cer']:.4f}")
        
        return results
