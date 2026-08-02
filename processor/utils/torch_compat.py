# torch_compat.py
"""
PyTorch compatibility patches for version 2.6+.
Fixes weights_only parameter issues with older model checkpoints.
"""
import torch
from functools import wraps

_patched = False


def patch_torch_load() -> bool:
    global _patched
    if _patched:
        return False
    
    _original_load = torch.load
    
    @wraps(_original_load)
    def _lenient_load(*args, **kwargs):
        kwargs.pop('weights_only', None)
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    
    torch.load = _lenient_load
    _patched = True
    return True


def add_safe_globals() -> bool:
    try:
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
        return True
    except AttributeError:
        return False


def ensure_compatibility(verbose: bool = True) -> None:
    try:
        patch_applied = patch_torch_load()
        add_safe_globals()
        
        if verbose and patch_applied:
            pass
    except Exception as e:
        if verbose:
            pass
