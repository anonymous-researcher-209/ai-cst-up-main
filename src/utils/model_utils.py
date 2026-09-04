"""Model utility functions for loading and parameter counting"""

import torch


def count_parameters(model):
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def safe_from_pretrained(cls, model_id, *args, **kwargs):
    """
    Safely load model using from_pretrained with fallback mechanisms
    
    Args:
        cls: Model or Processor class with from_pretrained method
        model_id: Hugging Face model ID
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Loaded model or processor instance
    """
    # Check torch version for safetensors preference
    try:
        _torch_ver = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    except Exception:
        _torch_ver = (0, 0)
    
    # Check if trust_remote_code is set (don't interfere with it)
    trust_remote_code = kwargs.get('trust_remote_code', False)
    
    use_safetensors = False
    if _torch_ver < (2, 6) and not trust_remote_code:
        try:
            import safetensors  # noqa: F401
            use_safetensors = True
        except ImportError:
            pass
    
    # Try with safetensors first if preferred (and not using trust_remote_code)
    if use_safetensors and not trust_remote_code:
        try:
            return cls.from_pretrained(model_id, *args, use_safetensors=True, **kwargs)
        except (TypeError, Exception):
            pass  # Fall back to default
    
    # Default attempt
    try:
        return cls.from_pretrained(model_id, *args, **kwargs)
    except TypeError:
        # Some loaders have different signatures
        return cls.from_pretrained(model_id)
