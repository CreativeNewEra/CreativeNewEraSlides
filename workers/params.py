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
    offload: bool = False
    t5_cpu: bool = False
    precision: str = "fp16"
