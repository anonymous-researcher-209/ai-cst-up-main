"""Base class for all ASR models"""

from abc import ABC, abstractmethod


class BaseASRModel(ABC):
    """Abstract base class for ASR models"""
    
    def __init__(self, name, model_id, model_type):
        """
        Initialize base model
        
        Args:
            name: Human-readable model name
            model_id: Model identifier (Hugging Face ID or API name)
            model_type: Type of model (whisper, ctc, qwen, api)
        """
        self.name = name
        self.model_id = model_id
        self.model_type = model_type
        self.model = None
        self.processor = None
    
    @abstractmethod
    def load(self):
        """
        Load model and processor
        
        Returns:
            dict: Model information (parameters, device, etc.)
        """
        pass
    
    @abstractmethod
    def transcribe(self, audio, audio_path=None):
        """
        Transcribe audio
        
        Args:
            audio: Processed audio tensor
            audio_path: Original audio file path (for some models)
            
        Returns:
            str: Transcription text
        """
        pass
    
    def cleanup(self):
        """Clean up model resources"""
        import torch
        
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        
        self.model = None
        self.processor = None
        
        # Free GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
