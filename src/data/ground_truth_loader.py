"""Ground truth data loader"""

import os

# Try to import script detection
try:
    from ..utils.transliteration import detect_script
    SCRIPT_DETECTION_AVAILABLE = True
except ImportError:
    SCRIPT_DETECTION_AVAILABLE = False


class GroundTruthLoader:
    """Load ground truth transcriptions from file with dual-script support"""
    
    def __init__(self, assets_path="assets", prefer_script=None):
        """
        Initialize GroundTruthLoader
        
        Args:
            assets_path: Path to assets directory containing grounds_truth.txt
            prefer_script: 'roman' or 'devanagari' - which script to use if dual-script available
        """
        self.assets_path = assets_path
        self.ground_truth_file = os.path.join(assets_path, "grounds_truth.txt")
        self.prefer_script = prefer_script  # None = auto-detect from model output
    
    def load(self, ground_truth_file=None, model_output_script=None):
        """
        Load ground truth transcriptions with dual-script support
        
        Args:
            ground_truth_file: Optional custom path to ground truth file
            model_output_script: 'roman' or 'devanagari' - select matching script (optional)
            
        Returns:
            dict: Mapping of audio filenames to transcriptions (str or dict with 'roman'/'devanagari' keys)
        """
        if ground_truth_file is None:
            ground_truth_file = self.ground_truth_file
        
        grounds_truths = {}
        
        if not os.path.exists(ground_truth_file):
            print(f"⚠ Ground truth file not found: {ground_truth_file}")
            return grounds_truths
        
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse line format: 
                # Single script: "audio_file.wav | transcription | diagnosis"
                # Dual script:   "audio_file.wav | roman_text | devanagari_text | diagnosis"
                if "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    audio_file = parts[0] if parts[0] else "audio.flac"
                    
                    # Detect format based on number of parts
                    if len(parts) == 2:
                        # Format: audio_file | transcription
                        transcription = parts[1]
                    elif len(parts) == 3:
                        # Could be: audio_file | transcription | diagnosis
                        # OR: audio_file | roman | devanagari
                        # Detect by checking if second part is Devanagari
                        if SCRIPT_DETECTION_AVAILABLE:
                            script1 = detect_script(parts[1])
                            script2 = detect_script(parts[2])
                            
                            if script1 == 'roman' and script2 == 'devanagari':
                                # Dual script format: audio | roman | devanagari
                                roman_text = parts[1]
                                devanagari_text = parts[2]
                                if model_output_script:
                                    # Specific script requested
                                    transcription = self._select_script(
                                        roman_text, devanagari_text, model_output_script
                                    )
                                else:
                                    # Return both versions for runtime selection
                                    transcription = {
                                        'roman': roman_text,
                                        'devanagari': devanagari_text
                                    }
                            else:
                                # Single script with diagnosis: audio | text | diagnosis
                                transcription = parts[1]
                        else:
                            # No script detection, assume single script format
                            transcription = parts[1]
                    elif len(parts) >= 4:
                        # Dual script with diagnosis: audio | roman | devanagari | diagnosis
                        roman_text = parts[1]
                        devanagari_text = parts[2]
                        if model_output_script:
                            # Specific script requested
                            transcription = self._select_script(
                                roman_text, devanagari_text, model_output_script
                            )
                        else:
                            # Return both versions for runtime selection
                            transcription = {
                                'roman': roman_text,
                                'devanagari': devanagari_text
                            }
                    else:
                        transcription = parts[1] if len(parts) > 1 else ""
                else:
                    # Default filename if not specified
                    audio_file = "audio.flac"
                    transcription = line
                
                # Make audio path relative to assets folder
                audio_path = os.path.join(self.assets_path, audio_file)
                grounds_truths[audio_path] = transcription
        
        return grounds_truths
    
    def _select_script(self, roman_text, devanagari_text, model_output_script):
        """
        Select appropriate script based on model output or preference.
        
        Args:
            roman_text: Roman script version
            devanagari_text: Devanagari script version
            model_output_script: Detected script from model output
            
        Returns:
            str: Selected transcription
        """
        # If model output script is specified, use matching ground truth
        if model_output_script == 'devanagari':
            return devanagari_text
        elif model_output_script == 'roman':
            return roman_text
        
        # Use preference if set
        if self.prefer_script == 'devanagari':
            return devanagari_text
        elif self.prefer_script == 'roman':
            return roman_text
        
        # Default to roman if no preference
        return roman_text
    
    def get_audio_files(self):
        """
        Get list of audio files in assets directory
        
        Returns:
            list: List of audio file paths
        """
        audio_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
        audio_files = []
        
        if not os.path.exists(self.assets_path):
            return audio_files
        
        for file in os.listdir(self.assets_path):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(self.assets_path, file))
        
        return audio_files
