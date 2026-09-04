"""CTC model implementation (e.g., IndicWav2Vec)"""

import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC
from .base_model import BaseASRModel
from ..utils.model_utils import count_parameters, safe_from_pretrained
from ..utils.device_utils import get_device


class CTCModel(BaseASRModel):
    """CTC-based ASR model (e.g., IndicWav2Vec)"""
    
    def __init__(self, name, model_id):
        """
        Initialize CTC model
        
        Args:
            name: Model name
            model_id: Hugging Face model ID
        """
        super().__init__(name, model_id, "ctc")
        self.device = get_device()
    
    def load(self):
        """Load CTC model and processor"""
        print(f"⚙ Loading CTC model...")
        
        # Load processor
        self.processor = safe_from_pretrained(AutoProcessor, self.model_id)
        print(f"✓ Processor loaded")
        
        # Load model
        self.model = safe_from_pretrained(AutoModelForCTC, self.model_id)
        print(f"✓ Model loaded")
        
        # Move to device
        if self.device == "cuda":
            try:
                self.model = self.model.to(self.device)
                print(f"✓ Model moved to GPU")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠ GPU OOM - Model will stay on CPU")
                    self.device = "cpu"
                else:
                    raise
        
        self.model.eval()
        
        # Get model info
        total_params, _ = count_parameters(self.model)
        params_millions = total_params / 1e6
        
        return {
            'parameters_M': params_millions,
            'device': str(next(self.model.parameters()).device),
            'total_parameters': total_params
        }
    
    def transcribe(self, audio, audio_path=None):
        """
        Transcribe audio using CTC model
        
        Args:
            audio: Processed audio tensor [1, N]
            audio_path: Not used for CTC
            
        Returns:
            str: Transcription
        """
        try:
            audio_array = audio.squeeze().numpy()
            
            # Process audio
            inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt")
            
            # Move to device
            if self.device == "cuda":
                input_values = inputs["input_values"].to(self.device)
            else:
                input_values = inputs["input_values"]
            
            # Get logits
            with torch.no_grad():
                logits = self.model(input_values).logits
                logits_np = logits[0].detach().cpu().numpy()
            
            # Decode
            if hasattr(self.processor, "decode"):
                # Use LM decoder if available
                decoded = self.processor.decode(logits_np)
                text = decoded.text if hasattr(decoded, "text") else decoded
            else:
                # Simple argmax decoding
                pred_ids = np.argmax(logits_np, axis=-1)
                text = self.processor.batch_decode([pred_ids])[0]
            
            return text.strip()
            
        except Exception as e:
            print(f"  ⚠ Error in CTC transcription: {e}")
            return ""
    
    def transcribe_batch(self, audio_batch):
        """
        Transcribe multiple audio files in a batch
        
        Args:
            audio_batch: Tensor of shape [B, max_length] - batched audio
            
        Returns:
            list: List of transcriptions
        """
        try:
            # Convert batch to numpy
            audio_arrays = [audio.numpy() for audio in audio_batch]
            
            # Process batch
            inputs = self.processor(
                audio_arrays,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            if self.device == "cuda":
                input_values = inputs["input_values"].to(self.device)
            else:
                input_values = inputs["input_values"]
            
            # Get logits for batch
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Decode each item in batch
            transcriptions = []
            for i in range(logits.shape[0]):
                logits_np = logits[i].detach().cpu().numpy()
                
                if hasattr(self.processor, "decode"):
                    decoded = self.processor.decode(logits_np)
                    text = decoded.text if hasattr(decoded, "text") else decoded
                else:
                    pred_ids = np.argmax(logits_np, axis=-1)
                    text = self.processor.batch_decode([pred_ids])[0]
                
                transcriptions.append(text.strip())
            
            return transcriptions
            
        except Exception as e:
            print(f"  ⚠ Error in batch CTC transcription: {e}")
            return [""] * len(audio_batch)
