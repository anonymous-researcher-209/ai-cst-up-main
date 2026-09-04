"""
Transliteration utilities for Hindi ASR evaluation.

Provides functions to convert between Devanagari and Roman scripts
for fair WER/CER comparison when ground truth and predictions use different scripts.
"""

import re
from typing import Optional, Tuple


# Devanagari to Roman mapping (ITRANS-like scheme)
DEVANAGARI_TO_ROMAN = {
    # Vowels
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऋ': 'ri',
    
    # Vowel marks (matras)
    'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ृ': 'ri',
    
    # Consonants
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
    'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
    'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
    'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va', 'श': 'sha',
    'ष': 'sha', 'स': 'sa', 'ह': 'ha',
    
    # Special consonants
    'क्ष': 'ksha', 'त्र': 'tra', 'ज्ञ': 'gya',
    'ड़': 'da', 'ढ़': 'dha', 'फ़': 'fa', 'ज़': 'za', 'ऱ': 'ra',
    
    # Halant (virama) - removes inherent 'a'
    '्': '',
    
    # Anusvara and Visarga
    'ं': 'n', 'ः': 'h', 'ँ': 'n',
    
    # Numerals
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    
    # Punctuation
    '।': '.', '॥': '.',
}

# Build reverse mapping for Roman to Devanagari
ROMAN_TO_DEVANAGARI = {}
for dev, rom in DEVANAGARI_TO_ROMAN.items():
    if rom and rom not in ROMAN_TO_DEVANAGARI:
        ROMAN_TO_DEVANAGARI[rom] = dev


def is_devanagari(text: str) -> bool:
    """
    Check if text contains Devanagari characters.
    
    Args:
        text: Input text
        
    Returns:
        True if text contains Devanagari characters
    """
    # Devanagari Unicode range: U+0900 to U+097F
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    return bool(devanagari_pattern.search(text))


def is_roman(text: str) -> bool:
    """
    Check if text is primarily Roman/Latin script.
    
    Args:
        text: Input text
        
    Returns:
        True if text is primarily Roman script
    """
    # Check if majority of alphabetic chars are ASCII
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return True
    ascii_chars = [c for c in alpha_chars if ord(c) < 128]
    return len(ascii_chars) / len(alpha_chars) > 0.5


def detect_script(text: str) -> str:
    """
    Detect the primary script of the text.
    
    Args:
        text: Input text
        
    Returns:
        'devanagari', 'roman', or 'mixed'
    """
    has_devanagari = is_devanagari(text)
    has_roman = is_roman(text)
    
    if has_devanagari and not has_roman:
        return 'devanagari'
    elif has_roman and not has_devanagari:
        return 'roman'
    else:
        return 'mixed'


def devanagari_to_roman(text: str) -> str:
    """
    Convert Devanagari text to Roman/Latin script.
    
    Uses a simplified ITRANS-like transliteration scheme.
    
    Args:
        text: Devanagari text
        
    Returns:
        Romanized text
    """
    result = []
    i = 0
    text_len = len(text)
    
    while i < text_len:
        char = text[i]
        
        # Check for conjuncts (2-char combinations)
        if i + 1 < text_len:
            two_char = text[i:i+2]
            if two_char in DEVANAGARI_TO_ROMAN:
                result.append(DEVANAGARI_TO_ROMAN[two_char])
                i += 2
                continue
        
        # Single character mapping
        if char in DEVANAGARI_TO_ROMAN:
            roman = DEVANAGARI_TO_ROMAN[char]
            
            # Handle consonant + halant (remove inherent 'a')
            if i + 1 < text_len and text[i + 1] == '्':
                # Remove trailing 'a' from consonant
                if roman.endswith('a') and len(roman) > 1:
                    roman = roman[:-1]
                result.append(roman)
                i += 2  # Skip halant
                continue
            
            # Handle consonant + matra
            if i + 1 < text_len and text[i + 1] in 'ािीुूेैोौृ':
                # Remove inherent 'a' before adding matra sound
                if roman.endswith('a') and len(roman) > 1:
                    roman = roman[:-1]
                result.append(roman)
                i += 1
                continue
            
            result.append(roman)
        else:
            # Keep non-Devanagari characters as-is
            result.append(char)
        
        i += 1
    
    # Clean up output
    output = ''.join(result)
    
    # Remove double vowels that might occur
    output = re.sub(r'aa+', 'aa', output)
    output = re.sub(r'ee+', 'ee', output)
    output = re.sub(r'oo+', 'oo', output)
    
    return output


def normalize_text(text: str, target_script: str = 'roman') -> str:
    """
    Normalize text to a target script for comparison.
    
    Args:
        text: Input text (can be Devanagari, Roman, or mixed)
        target_script: 'roman' or 'devanagari'
        
    Returns:
        Normalized text in target script
    """
    if target_script == 'roman':
        if is_devanagari(text):
            text = devanagari_to_roman(text)
        # Lowercase and normalize spaces
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    return text


def normalize_for_comparison(ground_truth: str, prediction: str) -> Tuple[str, str]:
    """
    Normalize both ground truth and prediction to the same script for fair comparison.
    
    Automatically detects scripts and converts to Roman for comparison.
    
    Args:
        ground_truth: Ground truth transcription
        prediction: Model prediction
        
    Returns:
        Tuple of (normalized_ground_truth, normalized_prediction)
    """
    gt_script = detect_script(ground_truth)
    pred_script = detect_script(prediction)
    
    # Normalize both to Roman (most common ground truth format)
    gt_normalized = normalize_text(ground_truth, 'roman')
    pred_normalized = normalize_text(prediction, 'roman')
    
    return gt_normalized, pred_normalized


def get_script_info(ground_truth: str, prediction: str) -> dict:
    """
    Get script information for ground truth and prediction.
    
    Args:
        ground_truth: Ground truth transcription
        prediction: Model prediction
        
    Returns:
        Dictionary with script detection info
    """
    return {
        'ground_truth_script': detect_script(ground_truth),
        'prediction_script': detect_script(prediction),
        'scripts_match': detect_script(ground_truth) == detect_script(prediction),
    }


# Try to use indic-transliteration library if available (more accurate)
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    
    def devanagari_to_roman_accurate(text: str) -> str:
        """
        Convert Devanagari to Roman using indic-transliteration library.
        More accurate than the basic mapping.
        """
        try:
            return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
        except Exception:
            # Fallback to basic implementation
            return devanagari_to_roman(text)
    
    # Override with better implementation
    _basic_devanagari_to_roman = devanagari_to_roman
    devanagari_to_roman = devanagari_to_roman_accurate
    
    TRANSLITERATION_LIBRARY = 'indic-transliteration'
    
except ImportError:
    TRANSLITERATION_LIBRARY = 'basic'


def get_transliteration_info() -> dict:
    """Get information about the transliteration backend."""
    return {
        'library': TRANSLITERATION_LIBRARY,
        'accurate': TRANSLITERATION_LIBRARY == 'indic-transliteration',
    }
