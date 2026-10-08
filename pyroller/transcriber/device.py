from __future__ import annotations

def auto_detect_transcriber_device() -> tuple[str | None, str | None]:
    """Return (device, compute_type) if a CUDA GPU is available, otherwise (None, None)."""
    try:
        import torch
    except ImportError:
        return None, None
    try:
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return None, None
