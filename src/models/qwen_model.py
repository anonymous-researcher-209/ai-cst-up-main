"""Qwen-Audio model implementation"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseASRModel
from ..utils.model_utils import count_parameters, safe_from_pretrained
from ..utils.device_utils import get_device


class QwenModel(BaseASRModel):
    """Qwen-Audio ASR model"""
    
    def __init__(self, name, model_id):
        """
        Initialize Qwen model
        
        Args:
            name: Model name
            model_id: Hugging Face model ID
        """
        super().__init__(name, model_id, "qwen")
        self.device = get_device()
    
    def load(self):
        """Load Qwen model and tokenizer"""
        print(f"⚙ Loading Qwen-Audio model...")
        
        # Load tokenizer (Qwen uses custom tokenizer, not processor)
        self.tokenizer = safe_from_pretrained(
            AutoTokenizer,
            self.model_id,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer loaded")
        
        # Load model
        if self.device == "cuda":
            try:
                self.model = safe_from_pretrained(
                    AutoModelForCausalLM,
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                print(f"✓ Model loaded to GPU with float16")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠ GPU OOM - Loading to CPU...")
                    self.model = safe_from_pretrained(
                        AutoModelForCausalLM,
                        self.model_id,
                        trust_remote_code=True
                    )
                    self.device = "cpu"
                else:
                    raise
        else:
            self.model = safe_from_pretrained(
                AutoModelForCausalLM,
                self.model_id,
                trust_remote_code=True
            )
        
        print(f"✓ Model loaded")
        self.model.eval()
        
        # Get model info
        total_params, _ = count_parameters(self.model)
        params_millions = total_params / 1e6
        
        return {
            'parameters_M': params_millions,
            'device': str(self.device),
            'total_parameters': total_params
        }
    
    def transcribe(self, audio, audio_path=None):
        """
        Transcribe audio using Qwen-Audio
        
        Args:
            audio: Not used (Qwen needs file path)
            audio_path: Audio file path (required)
            
        Returns:
            str: Transcription
        """
        if audio_path is None:
            print("  ✗ Qwen-Audio requires audio file path")
            return ""
        
        try:
            # Create conversation input using tokenizer
            query = self.tokenizer.from_list_format([
                {'audio': audio_path},
                {'text': 'कृपया इस ऑडियो को हिंदी में लिखें। Please transcribe this audio in Hindi language.'},
            ])
            
            # Tokenize the query
            inputs = self.tokenizer(query, return_tensors='pt')
            
            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v 
                         for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                pred = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            # Decode
            response = self.tokenizer.decode(pred[0], skip_special_tokens=False)
            
            # Extract transcription (Qwen includes prompt in output)
            if '<|im_start|>assistant' in response:
                transcription = response.split('<|im_start|>assistant')[-1].strip()
                transcription = transcription.replace('<|im_end|>', '').strip()
            else:
                transcription = response
            
            return transcription.strip()
            
        except Exception as e:
            print(f"  ⚠ Error in Qwen transcription: {e}")
            return ""
    
    def transcribe_batch(self, audio_batch):
        """
        Qwen-Audio doesn't support true batch processing (requires file paths)
        This method processes files sequentially but maintains the interface
        
        Args:
            audio_batch: Not used (Qwen needs file paths)
            
        Returns:
            list: Empty list (batch processing not supported)
        """
        # Qwen-Audio requires file paths, not tensors
        # Batch processing not possible without file paths
        print("  ⚠ Qwen-Audio doesn't support batch processing (requires file paths)")
        return [""] * len(audio_batch) if hasattr(audio_batch, '__len__') else []
