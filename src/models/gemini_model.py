"""Gemini Audio API model implementation"""

import os
import time
from .base_model import BaseASRModel


class GeminiModel(BaseASRModel):
    """Gemini Audio API model"""
    
    def __init__(self, name="Gemini Audio", model_id="gemini-1.5-flash"):
        """
        Initialize Gemini model
        
        Args:
            name: Model name
            model_id: Gemini model identifier
        """
        super().__init__(name, model_id, "gemini")
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.genai = None
        self.model = None
    
    def load(self):
        """Initialize Gemini API"""
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        print(f"✓ Gemini API key found (ends with: ...{self.api_key[-4:]})")
        
        # Configure API
        self.genai.configure(api_key=self.api_key)
        self.model = self.genai.GenerativeModel(self.model_id)
        
        return {
            'parameters_M': 0,  # API model, parameters unknown
            'device': 'api',
            'total_parameters': 0
        }
    
    def transcribe(self, audio, audio_path=None):
        """
        Transcribe audio using Gemini API
        
        Args:
            audio: Not used (API needs file path)
            audio_path: Audio file path (required)
            
        Returns:
            str: Transcription
        """
        if audio_path is None:
            print("  ✗ Gemini requires audio file path")
            return ""
        
        try:
            # Upload audio file
            audio_file = self.genai.upload_file(path=audio_path)
            
            # Generate transcription
            response = self.model.generate_content([
                "कृपया इस ऑडियो को हिंदी में लिखें। Please transcribe this audio in Hindi language accurately.",
                audio_file
            ])
            
            return response.text.strip()
            
        except Exception as e:
            print(f"  ⚠ Error in Gemini transcription: {e}")
            return ""
    
    def transcribe_batch(self, audio_batch):
        """
        Gemini API doesn't support true batch processing
        This method maintains the interface but batch processing not applicable
        
        Args:
            audio_batch: Not used (API requires file paths)
            
        Returns:
            list: Empty list (batch processing not supported)
        """
        # Gemini API requires file paths and makes individual API calls
        # Batch processing not applicable for API-based models
        print("  ⚠ Gemini API doesn't support batch processing")
        return [""] * len(audio_batch) if hasattr(audio_batch, '__len__') else []
    
    def cleanup(self):
        """Clean up API resources"""
        self.model = None
        self.genai = None
