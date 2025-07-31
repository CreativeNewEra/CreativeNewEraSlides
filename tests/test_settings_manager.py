import importlib
import pathlib
import sys
import types

# Stub QSettings with in-memory dictionary
class FakeQSettings:
    def __init__(self, *args, **kwargs):
        self.store = {}
    def value(self, key, default=None):
        return self.store.get(key, default)
    def setValue(self, key, value):
        self.store[key] = value

pyqt5 = types.ModuleType("PyQt5")
qtcore = types.ModuleType("PyQt5.QtCore")
qtcore.QSettings = FakeQSettings
sys.modules["PyQt5"] = pyqt5
sys.modules["PyQt5.QtCore"] = qtcore

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

settings_manager = importlib.reload(importlib.import_module("utils.settings_manager"))


def test_get_and_set():
    sm = settings_manager.SettingsManager()
    sm.set("foo", "bar")
    assert sm.get("foo") == "bar"


def test_model_device_and_output_dir():
    sm = settings_manager.SettingsManager()
    sm.set_model_path("flux", "/model")
    assert sm.get_model_path("flux") == "/model"
    sm.set_device("cuda")
    assert sm.get_device() == "cuda"
    sm.set_output_dir("/tmp")
    assert sm.get_output_dir() == "/tmp"
