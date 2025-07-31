from PyQt5.QtCore import QSettings
from diffusers import StableDiffusionPipeline
import torch


class SettingsManager:
    """
    Wrapper around QSettings to persist application settings.
    """
    def __init__(self):
        # Organization and Application name determine registry/storage keys
        self._q = QSettings("YourCompany", "FluxWanApp")

    def get(self, key: str, default=None):
        return self._q.value(key, default)

    def set(self, key: str, value):
        self._q.setValue(key, value)

    def get_model_path(self, key: str, default: str = ""):  # e.g. "flux" or "wan"
        return self.get(f"models/{key}", default)

    def set_model_path(self, key: str, path: str):
        self.set(f"models/{key}", path)

    def get_device(self) -> str:
        return self.get("device", "cpu")

    def set_device(self, device: str):
        self.set("device", device)

    def get_output_dir(self, default: str = ".") -> str:
        return self.get("output_dir", default)

    def set_output_dir(self, path: str):
        self.set("output_dir", path)


class ModelManager:
    """
    Caches and returns loaded models/pipelines.
    """
    _flux_pipe = None

    @classmethod
    def get_flux_pipeline(cls, params: dict):
        if cls._flux_pipe is None:
            model_path = params.get('model_path') or SettingsManager().get_model_path('flux')
            device = SettingsManager().get_device()
            # Load with mixed precision if possible
            dtype = torch.float16 if device != "cpu" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )
            pipe.to(device)
            cls._flux_pipe = pipe
        return cls._flux_pipe
