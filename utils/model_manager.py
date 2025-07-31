from diffusers import StableDiffusionPipeline
import torch

from .settings_manager import SettingsManager


class ModelManager:
    """Caches and returns loaded models/pipelines."""

    _flux_pipe = None

    @classmethod
    def get_flux_pipeline(cls, params: dict):
        if cls._flux_pipe is None:
            model_path = params.get('model_path') or SettingsManager().get_model_path('flux')
            device = SettingsManager().get_device()
            dtype = torch.float16 if device != "cpu" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )
            pipe.to(device)
            cls._flux_pipe = pipe
        return cls._flux_pipe
