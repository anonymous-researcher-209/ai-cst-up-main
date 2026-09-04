"""Audio file processing and normalization"""

import torch
import soundfile as sf
import torchaudio
from torch.nn.utils.rnn import pad_sequence


class AudioProcessor:
    """Process audio files for ASR models"""
    
    def __init__(self, target_sample_rate=16000):
        """
        Initialize AudioProcessor
        
        Args:
            target_sample_rate: Target sample rate for audio (default: 16000 Hz)
        """
        self.target_sample_rate = target_sample_rate
    
    def load_and_process(self, file_path):
        """
        Load and process audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            torch.Tensor: Processed audio waveform [1, N] or None if error
        """
        try:
            # Load audio using soundfile
            waveform, sample_rate = sf.read(file_path)
            
            # Convert to torch tensor
            waveform = torch.from_numpy(waveform).float()
            
            # Convert stereo to mono
            if len(waveform.shape) > 1:
                waveform = torch.mean(waveform, dim=1)
            
            # Add channel dimension
            waveform = waveform.unsqueeze(0)
            
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                waveform = self._resample(waveform, sample_rate, self.target_sample_rate)
            
            # Normalize
            waveform = self._normalize(waveform)
            
            return waveform
            
        except Exception as e:
            print(f"Error processing audio {file_path}: {e}")
            return None
    
    def _resample(self, waveform, orig_freq, new_freq):
        """Resample audio to target frequency"""
        resampler = torchaudio.transforms.Resample(orig_freq, new_freq)
        return resampler(waveform)
    
    def _normalize(self, waveform):
        """Normalize audio waveform to [-1, 1]"""
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            return waveform / max_val
        return waveform    
    def load_batch(self, file_paths):
        """
        Load and process multiple audio files as a batch
        
        Args:
            file_paths: List of paths to audio files
            
        Returns:
            tuple: (list of waveforms, list of valid file paths, list of failed paths)
        """
        waveforms = []
        valid_paths = []
        failed_paths = []
        
        for file_path in file_paths:
            audio = self.load_and_process(file_path)
            if audio is not None:
                waveforms.append(audio.squeeze(0))  # Remove batch dim for padding
                valid_paths.append(file_path)
            else:
                failed_paths.append(file_path)
        
        return waveforms, valid_paths, failed_paths
    
    def pad_batch(self, waveforms):
        """
        Pad waveforms to same length for batching
        
        Args:
            waveforms: List of 1D tensors
            
        Returns:
            torch.Tensor: Padded batch [B, max_length]
        """
        if not waveforms:
            return torch.empty(0)
        
        # Pad sequences to same length
        padded = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
        return padded