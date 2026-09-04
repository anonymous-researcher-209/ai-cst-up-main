"""Whisper model implementation"""

import warnings
import torch
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from .base_model import BaseASRModel
from ..utils.model_utils import count_parameters, safe_from_pretrained
from ..utils.device_utils import get_device

# Suppress known harmless warnings from transformers
warnings.filterwarnings("ignore", message=".*SuppressTokensLogitsProcessor.*")
warnings.filterwarnings("ignore", message=".*SuppressTokensAtBeginLogitsProcessor.*")
warnings.filterwarnings("ignore", message=".*generation config is outdated.*")
warnings.filterwarnings("ignore", message=".*custom `forced_decoder_ids`.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")

# Suppress transformers generation warnings
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)


class WhisperModel(BaseASRModel):
    """Whisper-based ASR model"""
    
    def __init__(self, name, model_id, language="hindi"):
        """
        Initialize Whisper model
        
        Args:
            name: Model name
            model_id: Hugging Face model ID
            language: Target language for transcription
        """
        super().__init__(name, model_id, "whisper")
        self.language = language
        self.device = get_device()
        self.use_fp16 = False  # Track if using FP16
        self.use_forced_decoder_ids = False  # Track if using legacy API
    
    def load(self):
        """Load Whisper model and processor"""
        print(f"⚙ Loading Whisper model...")
        
        # Load processor
        self.processor = safe_from_pretrained(WhisperProcessor, self.model_id)
        print(f"✓ Processor loaded")
        
        # Load model
        self.model = safe_from_pretrained(WhisperForConditionalGeneration, self.model_id)
        print(f"✓ Model loaded")
        
        # Move to device with optimizations
        if self.device == "cuda":
            try:
                self.model = self.model.to(self.device)
                # Enable half precision for faster inference
                self.model = self.model.half()
                self.use_fp16 = True
                print(f"✓ Model moved to GPU (FP16 enabled)")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠ GPU OOM - Model will stay on CPU")
                    self.device = "cpu"
                    self.use_fp16 = False
                else:
                    raise
        
        self.model.eval()
        
        # Check if model supports modern generation config
        # Some models (like AI4Bharat) have outdated configs
        try:
            # Test if the model supports language parameter
            test_gen_config = self.model.generation_config.to_dict()
            
            # Check if this is an older model that needs forced_decoder_ids
            # AI4Bharat and some fine-tuned models have outdated generation configs
            needs_legacy_api = False
            
            # If config has forced_decoder_ids but no lang_to_id, it's an older model
            if 'forced_decoder_ids' in test_gen_config:
                needs_legacy_api = True
            
            # Check for AI4Bharat/vasista models which need legacy API
            if 'ai4bharat' in self.model_id.lower() or 'vasista' in self.model_id.lower():
                needs_legacy_api = True
                
            if needs_legacy_api:
                self.use_forced_decoder_ids = True
                print(f"  ℹ️ Using legacy decoder IDs (older model config)")
            else:
                # Try to update generation config for modern API
                self.model.generation_config.task = "transcribe"
                self.model.generation_config.language = self.language
                print(f"  ℹ️ Using modern generation config")
        except Exception as e:
            # Fall back to forced_decoder_ids
            self.use_forced_decoder_ids = True
            print(f"  ℹ️ Using legacy decoder IDs (fallback: {e})")
        
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
        Transcribe audio using Whisper
        
        Args:
            audio: Processed audio tensor [1, N]
            audio_path: Not used for Whisper
            
        Returns:
            str: Transcription
        """
        try:
            # Suppress generation warnings from transformers
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                audio_array = audio.squeeze().numpy()
                
                # Process audio
                inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt")
                
                # Move to device and convert to FP16 if needed
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    if self.use_fp16:
                        inputs["input_features"] = inputs["input_features"].half()
                
                # Generation kwargs - optimized for speed
                generate_kwargs = {
                    "max_new_tokens": 256,  # Sufficient for ~20s audio
                    "max_length": None,  # Explicitly set to None to avoid conflict
                    "num_beams": 1,  # Greedy decoding (3x faster than beam search)
                    "do_sample": False,
                    "use_cache": True,
                }
                
                # Use appropriate API based on model compatibility
                if self.use_forced_decoder_ids:
                    # Legacy API for older models (AI4Bharat, etc.)
                    # Set language in generation config instead of passing as parameter
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=self.language,
                        task="transcribe"
                    )
                    # Update generation config to include forced_decoder_ids
                    self.model.generation_config.forced_decoder_ids = forced_decoder_ids
                else:
                    # Modern API
                    generate_kwargs["language"] = self.language
                    generate_kwargs["task"] = "transcribe"
                
                # Generate transcription
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        inputs["input_features"],
                        **generate_kwargs
                    )
                
                # Decode
                transcription = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]
                
                return transcription.strip()
            
        except Exception as e:
            print(f"  ⚠ Error in Whisper transcription: {e}")
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
            # Suppress generation warnings from transformers
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Convert batch to numpy
                audio_arrays = [audio.numpy() for audio in audio_batch]
                
                # Process batch
                inputs = self.processor(
                    audio_arrays, 
                    sampling_rate=16000, 
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True
                )
                
                # Move to device and convert to FP16 if needed
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    if self.use_fp16:
                        inputs["input_features"] = inputs["input_features"].half()
                
                # Generation kwargs - balanced speed and quality
                generate_kwargs = {
                    "max_new_tokens": 256,  # Sufficient for ~20s audio
                    "max_length": None,  # Explicitly set to None to avoid conflict
                    "num_beams": 2,  # Balanced beam search
                    "do_sample": False,
                    "use_cache": True,
                }
                
                # Use appropriate API based on model compatibility
                if self.use_forced_decoder_ids:
                    # Legacy API for older models (AI4Bharat, etc.)
                    # Set language in generation config instead of passing as parameter
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=self.language,
                        task="transcribe"
                    )
                    # Update generation config to include forced_decoder_ids
                    self.model.generation_config.forced_decoder_ids = forced_decoder_ids
                else:
                    # Modern API
                    generate_kwargs["language"] = self.language
                    generate_kwargs["task"] = "transcribe"
                
                # Generate transcriptions
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        inputs["input_features"],
                        attention_mask=inputs.get("attention_mask"),
                        **generate_kwargs
                    )
                
                # Decode all transcriptions
                transcriptions = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )
                
                return [t.strip() for t in transcriptions]
            
        except Exception as e:
            print(f"  ⚠ Error in batch Whisper transcription: {e}")
            return [""] * len(audio_batch)
