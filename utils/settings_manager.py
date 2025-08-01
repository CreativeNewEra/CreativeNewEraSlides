from typing import Any, Optional

from PyQt5.QtCore import QSettings


class SettingsManager:
    """Wrapper around :class:`QSettings` to persist application settings."""

    def __init__(self) -> None:
        """Initialize the underlying :class:`QSettings` store."""
        # Organization and Application name determine registry/storage keys
        self._q = QSettings("YourCompany", "FluxWanApp")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Return the stored value for ``key`` or ``default`` if missing."""
        return self._q.value(key, default)

    def set(self, key: str, value: Any) -> None:
        """Persist ``value`` under ``key``."""
        self._q.setValue(key, value)

    def get_model_path(self, key: str, default: str = "") -> str:
        """Return model path for ``key`` such as ``"flux"`` or ``"wan"``."""
        return self.get(f"models/{key}", default)

    def set_model_path(self, key: str, path: str) -> None:
        """Store ``path`` as the model location for ``key``."""
        self.set(f"models/{key}", path)

    def get_device(self) -> str:
        """Retrieve the last-used compute device."""
        return self.get("device", "cpu")

    def set_device(self, device: str) -> None:
        """Persist the selected compute ``device``."""
        self.set("device", device)

    def get_output_dir(self, default: str = ".") -> str:
        """Return the directory for generated output files."""
        return self.get("output_dir", default)

    def set_output_dir(self, path: str) -> None:
        """Persist the directory for generated output files."""
        self.set("output_dir", path)
