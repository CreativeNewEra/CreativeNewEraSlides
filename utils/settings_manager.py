from PyQt5.QtCore import QSettings


class SettingsManager:
    """Wrapper around QSettings to persist application settings."""

    def __init__(self):
        # Organization and Application name determine registry/storage keys
        self._q = QSettings("YourCompany", "FluxWanApp")

    def get(self, key: str, default=None):
        return self._q.value(key, default)

    def set(self, key: str, value):
        self._q.setValue(key, value)

    def get_model_path(self, key: str, default: str = ""):
        """Return stored model path, e.g. for 'flux' or 'wan'."""
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
