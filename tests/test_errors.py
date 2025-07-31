import importlib
import pathlib
import sys
import types


# Provide a minimal torch stub so utils.errors can be imported without the real
# heavy dependency.
fake_torch = types.ModuleType("torch")


class DummyOOM(RuntimeError):
    pass


setattr(fake_torch, "cuda", types.SimpleNamespace(OutOfMemoryError=DummyOOM))
sys.modules.setdefault("torch", fake_torch)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

errors = importlib.import_module("utils.errors")


def test_file_not_found_message():
    msg = errors.parse_error(FileNotFoundError("missing.txt"))
    assert "Required file not found" in msg


def test_cuda_out_of_memory_message():
    exc = fake_torch.cuda.OutOfMemoryError("CUDA out of memory")
    msg = errors.parse_error(exc)
    assert "CUDA out of memory" in msg


def test_os_error_message():
    msg = errors.parse_error(OSError("permission denied"))
    assert "OS error" in msg


def test_runtime_error_message():
    msg = errors.parse_error(RuntimeError("boom"))
    assert "Runtime error" in msg
