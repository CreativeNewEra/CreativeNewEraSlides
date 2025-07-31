import importlib
import pathlib
import sys
import types

# Stub torch with minimal attributes
class DummyOOM(RuntimeError):
    pass

fake_torch = types.SimpleNamespace(
    float16="float16",
    float32="float32",
    cuda=types.SimpleNamespace(OutOfMemoryError=DummyOOM)
)
sys.modules["torch"] = fake_torch

# Stub diffusers StableDiffusionPipeline
class FakePipe:
    def __init__(self):
        self.to_calls = []
    def to(self, device):
        self.to_calls.append(device)
        return self
    def __call__(self, *args, **kwargs):
        return types.SimpleNamespace(images=[])

class FakeStableDiffusionPipeline:
    from_pretrained_calls = []
    @classmethod
    def from_pretrained(cls, model_path, torch_dtype):
        cls.from_pretrained_calls.append((model_path, torch_dtype))
        return FakePipe()

fake_diffusers = types.ModuleType("diffusers")
fake_diffusers.StableDiffusionPipeline = FakeStableDiffusionPipeline
sys.modules.setdefault("diffusers", fake_diffusers)

# Stub PyQt5 for SettingsManager
pyqt5 = types.ModuleType("PyQt5")
qtcore = types.ModuleType("PyQt5.QtCore")
qtcore.QSettings = type(
    "QSettings", (), {"__init__": lambda self, *a, **k: None, "value": lambda self, k, d=None: d, "setValue": lambda self, k, v: None}
)
sys.modules.setdefault("PyQt5", pyqt5)
sys.modules.setdefault("PyQt5.QtCore", qtcore)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

model_manager = importlib.import_module("utils.model_manager")


def setup_function(function):
    model_manager.ModelManager._flux_pipe = None
    model_manager.ModelManager._flux_device = None
    model_manager.ModelManager._settings_manager = None
    FakeStableDiffusionPipeline.from_pretrained_calls = []


def test_loads_and_caches_pipeline():
    params = {"model_path": "dummy", "device": "cpu"}
    pipe1 = model_manager.ModelManager.get_flux_pipeline(params)
    pipe2 = model_manager.ModelManager.get_flux_pipeline(params)
    assert pipe1 is pipe2
    assert FakeStableDiffusionPipeline.from_pretrained_calls == [("dummy", "float32")]
    assert pipe1.to_calls == ["cpu"]


def test_moves_pipeline_to_new_device():
    params = {"model_path": "dummy", "device": "cpu"}
    pipe1 = model_manager.ModelManager.get_flux_pipeline(params)
    params2 = {"model_path": "dummy", "device": "cuda"}
    pipe2 = model_manager.ModelManager.get_flux_pipeline(params2)
    assert pipe1 is pipe2
    assert FakeStableDiffusionPipeline.from_pretrained_calls == [("dummy", "float32")]
    assert pipe1.to_calls == ["cpu", "cuda"]
