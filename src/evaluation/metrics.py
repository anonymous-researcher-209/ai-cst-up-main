"""Metrics calculation for ASR evaluation"""

from jiwer import wer, cer

# Import transliteration utilities
try:
    from ..utils.transliteration import (
        normalize_for_comparison, 
        detect_script, 
        is_devanagari,
        get_transliteration_info
    )
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False


def calculate_wer(reference, hypothesis, normalize_script=True):
    """
    Calculate Word Error Rate
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalize_script: If True, normalize both texts to same script before comparison
        
    Returns:
        float: WER score (0.0 = perfect, 1.0 = completely wrong)
    """
    try:
        # Normalize scripts for fair comparison
        if normalize_script and TRANSLITERATION_AVAILABLE:
            reference, hypothesis = normalize_for_comparison(reference, hypothesis)
        
        return wer(reference, hypothesis)
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return 1.0


def calculate_cer(reference, hypothesis, normalize_script=True):
    """
    Calculate Character Error Rate
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalize_script: If True, normalize both texts to same script before comparison
        
    Returns:
        float: CER score (0.0 = perfect, 1.0 = completely wrong)
    """
    try:
        # Normalize scripts for fair comparison
        if normalize_script and TRANSLITERATION_AVAILABLE:
            reference, hypothesis = normalize_for_comparison(reference, hypothesis)
        
        return cer(reference, hypothesis)
    except Exception as e:
        print(f"Error calculating CER: {e}")
        return 1.0


def calculate_metrics(reference, hypothesis, normalize_script=True):
    """
    Calculate all metrics with optional script normalization.
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalize_script: If True, normalize both texts to same script
        
    Returns:
        dict: Dictionary with wer, cer, and normalization info
    """
    result = {
        'wer': calculate_wer(reference, hypothesis, normalize_script),
        'cer': calculate_cer(reference, hypothesis, normalize_script),
        'normalized': normalize_script and TRANSLITERATION_AVAILABLE,
    }
    
    if TRANSLITERATION_AVAILABLE:
        result['reference_script'] = detect_script(reference)
        result['hypothesis_script'] = detect_script(hypothesis)
        result['scripts_matched'] = result['reference_script'] == result['hypothesis_script']
    
    return result


def get_metrics_info():
    """Get information about metrics configuration."""
    info = {
        'transliteration_available': TRANSLITERATION_AVAILABLE,
    }
    if TRANSLITERATION_AVAILABLE:
        info.update(get_transliteration_info())
    return info
