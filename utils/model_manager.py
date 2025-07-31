from diffusers import StableDiffusionPipeline
import logging
import threading
import torch

from .settings_manager import SettingsManager


logger = logging.getLogger(__name__)


class ModelManager:
    """Caches and returns loaded models/pipelines."""

    _flux_pipe = None
    _flux_device = None
    _settings_manager = None
    _flux_lock = threading.Lock()

    @classmethod
    def _get_settings_manager(cls):
        if cls._settings_manager is None:
            cls._settings_manager = SettingsManager()
        return cls._settings_manager

    @classmethod
    def get_flux_pipeline(cls, params: dict):
        settings_manager = cls._get_settings_manager()
        model_path = params.get('model_path') or settings_manager.get_model_path('flux')
        requested_device = params.get('device') or settings_manager.get_device()
        dtype = torch.float16 if requested_device != "cpu" else torch.float32

        with cls._flux_lock:
            if cls._flux_pipe is None:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                )
                pipe.to(requested_device)
                cls._flux_pipe = pipe
                cls._flux_device = requested_device
            elif cls._flux_device != requested_device:
                pipe = cls._flux_pipe
                try:
                    pipe.to(requested_device)
                except (RuntimeError, OSError):
                    logger.exception(
                        "Failed moving flux pipeline to %s; reloading", requested_device
                    )
                    try:
                        pipe = StableDiffusionPipeline.from_pretrained(
                            model_path,
                            torch_dtype=dtype,
                        )
                        pipe.to(requested_device)
                        cls._flux_pipe = pipe
                    except (RuntimeError, OSError) as load_exc:
                        raise RuntimeError(
                            f"Could not load flux pipeline on {requested_device}"
                        ) from load_exc
                cls._flux_device = requested_device

            return cls._flux_pipe
