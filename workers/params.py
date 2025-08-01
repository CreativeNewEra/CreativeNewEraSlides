from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageParams:
    """Parameters controlling image generation."""

    width: int
    height: int
    steps: int
    guidance: float
    # Path to the model file or directory used for image generation.
    model_path: Optional[str] = None
    device: str = "cpu"
    quantized: bool = False


@dataclass
class VideoParams:
    """Parameters controlling video generation."""

    width: int
    height: int
    frames: int
    steps: int
    offload: bool = (
        False  # Whether to offload model weights to CPU when not in use. Useful for memory-constrained environments.
    )
    t5_cpu: bool = (
        False  # If True, forces the T5 model to run on the CPU regardless of the main device setting.
    )
    precision: str = (
        "fp16"  # Numerical precision to use for computations. Valid values are "fp16" (half-precision) and "fp32" (full-precision).
    )
