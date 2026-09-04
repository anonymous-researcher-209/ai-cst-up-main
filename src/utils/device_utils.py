"""Device detection and management utilities"""

import torch


def get_device():
    """Get the best available device (CUDA or CPU)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device_info():
    """Get detailed device information"""
    device = get_device()
    info = {"device": device}
    
    if device == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info["gpu_count"] = torch.cuda.device_count()
    
    return info


def print_device_info():
    """Print device information to console"""
    info = get_device_info()
    print(f"🔧 Using device: {info['device']}")
    
    if info['device'] == "cuda":
        print(f"   GPU: {info['gpu_name']}")
        print(f"   VRAM: {info['vram_gb']:.2f} GB")
        if info['gpu_count'] > 1:
            print(f"   GPUs Available: {info['gpu_count']}")
