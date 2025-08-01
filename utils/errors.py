import torch


def parse_error(exc: Exception) -> str:
    """
    Interpret common exceptions and return user-friendly messages.
    """
    # File or path errors
    if isinstance(exc, FileNotFoundError):
        filename = exc.filename or "<unknown>"
        return f"Required file not found: {filename}. Please check your model paths."
    if isinstance(exc, OSError):
        # E.g., CLI not installed, permission denied
        return f"OS error: {exc.strerror or str(exc)}."

    # Torch / CUDA OOM
    msg = str(exc)
    if "out of memory" in msg.lower() or isinstance(exc, torch.cuda.OutOfMemoryError):
        return "CUDA out of memory. Try lowering resolution or steps, or switch to CPU."

    # Wan CLI failures
    if isinstance(exc, RuntimeError):
        return f"Runtime error: {msg}"

    # Fallback
    return f"Unexpected error: {msg}"
