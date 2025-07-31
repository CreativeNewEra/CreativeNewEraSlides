from diffusers import StableDiffusionPipeline
import torch

from .settings_manager import SettingsManager


class ModelManager:
    """Caches and returns loaded models/pipelines."""

    _flux_pipe = None
    _settings_manager = None

    @classmethod
    def _get_settings_manager(cls):
        if cls._settings_manager is None:
            cls._settings_manager = SettingsManager()
        return cls._settings_manager

    @classmethod
    def get_flux_pipeline(cls, params: dict):
        if cls._flux_pipe is None:
            settings_manager = cls._get_settings_manager()
            model_path = params.get('model_path') or settings_manager.get_model_path('flux')
            device = settings_manager.get_device()
            dtype = torch.float16 if device != "cpu" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )
            pipe.to(device)
            cls._flux_pipe = pipe
        return cls._flux_pipe
