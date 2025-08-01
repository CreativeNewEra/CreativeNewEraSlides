import importlib
import os
import pathlib
import sys
import types


# ---- Stubs for PyQt5 ----
class DummySignal:
    def __init__(self, *args, **kwargs):
        self.emitted = []

    def connect(self, func):
        self._func = func

    def emit(self, value):
        self.emitted.append(value)
        if hasattr(self, "_func"):
            self._func(value)


class QThread:
    def __init__(self, parent=None):
        self.parent = parent


class QImage:
    Format_RGBA8888 = 0

    def __init__(self, data, width, height, fmt):
        self.data = data
        self.width = width
        self.height = height
        self.fmt = fmt


pyqt5 = types.ModuleType("PyQt5")
qtcore = types.ModuleType("PyQt5.QtCore")
qtcore.QThread = QThread  # type: ignore[attr-defined]
qtcore.pyqtSignal = DummySignal  # type: ignore[attr-defined]
qtcore.QObject = object  # type: ignore[attr-defined]
qtgui = types.ModuleType("PyQt5.QtGui")
qtgui.QImage = QImage  # type: ignore[attr-defined]
sys.modules["PyQt5"] = pyqt5
sys.modules["PyQt5.QtCore"] = qtcore
sys.modules["PyQt5.QtGui"] = qtgui


# ---- Stub torch ----
class DummyOOM(RuntimeError):
    pass


def empty_cache():
    empty_cache.called = True  # type: ignore[attr-defined]


empty_cache.called = False  # type: ignore[attr-defined]

fake_torch = types.SimpleNamespace(
    float16="float16",
    float32="float32",
    cuda=types.SimpleNamespace(
        OutOfMemoryError=DummyOOM,
        empty_cache=empty_cache,
    ),
)
sys.modules["torch"] = fake_torch  # type: ignore[assignment]


# ---- Stub PIL ----
class DummyImage:
    mode = "RGBA"
    width = 1
    height = 1
    MINIMAL_IMAGE_DATA = b"00"  # Minimal image data for tests

    def convert(self, mode):
        return self

    def tobytes(self, *args):
        return self.MINIMAL_IMAGE_DATA


pil = types.ModuleType("PIL")
image_mod = types.ModuleType("PIL.Image")
image_mod.Image = DummyImage  # type: ignore[attr-defined]
pil.Image = image_mod  # type: ignore[attr-defined]
sys.modules.setdefault("PIL", pil)
sys.modules.setdefault("PIL.Image", image_mod)


# ---- Stub utils.model_manager ----
class FakePipeline:
    def __call__(self, **kwargs):
        callback = kwargs.get("callback")
        steps = kwargs.get("num_inference_steps", 0)
        for i in range(steps):
            if callback:
                callback(i, None, None)
        return types.SimpleNamespace(images=[DummyImage()])


fake_model_manager = types.ModuleType("utils.model_manager")
# noqa: E501 is used here because the type-ignore comment makes the line long
fake_model_manager.ModelManager = types.SimpleNamespace(  # type: ignore[attr-defined]  # noqa: E501
    get_flux_pipeline=lambda params: FakePipeline(),
)
sys.modules["utils.model_manager"] = fake_model_manager

# ---- Import workers module ----
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from workers.params import ImageParams, VideoParams
workers = importlib.import_module("workers.image_and_video_workers")


# ---- Tests for ImageWorker ----
def test_image_worker_emits_progress_and_result():
    empty_cache.called = False
    params = ImageParams(width=1, height=1, steps=2, guidance=1)
    worker = workers.ImageWorker("prompt", "", params)
    worker.progress = DummySignal()
    worker.result = DummySignal()
    worker.error = DummySignal()
    workers.ImageWorker.run(worker)
    assert worker.progress.emitted[-1] == 100
    assert len(worker.result.emitted) == 1
    assert worker.error.emitted == []
    assert empty_cache.called


def test_image_worker_passes_quantized_flag():
    captured_params = {}
    fake_model_manager.ModelManager.get_flux_pipeline = (
        lambda params: captured_params.update(params) or FakePipeline()
    )
    params = ImageParams(width=1, height=1, steps=1, guidance=1, quantized=True)
    worker = workers.ImageWorker("prompt", "", params)
    worker.progress = DummySignal()
    worker.result = DummySignal()
    worker.error = DummySignal()
    workers.ImageWorker.run(worker)
    assert captured_params.get("quantized") is True


def test_image_worker_emits_error_on_failure():
    class BadPipeline:
        def __call__(self, **kwargs):
            raise RuntimeError("boom")

    fake_model_manager.ModelManager.get_flux_pipeline = (
        lambda params: BadPipeline()
    )  # noqa: E501
    params = ImageParams(width=1, height=1, steps=1, guidance=1)
    worker = workers.ImageWorker("p", "", params)
    worker.progress = DummySignal()
    worker.result = DummySignal()
    worker.error = DummySignal()
    workers.ImageWorker.run(worker)
    assert worker.result.emitted == []
    assert worker.error.emitted and "Runtime error" in worker.error.emitted[0]


# ---- Tests for VideoWorker ----
def test_video_worker_builds_command_and_emits_progress(tmp_path):
    captured_cmd = []

    def fake_popen(cmd, stdout, stderr, text):
        captured_cmd.append(cmd)

        class Proc:
            returncode = 0
            stdout = ["Progress: 10%\n", "Progress: 100%\n"]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def kill(self):
                pass

        return Proc()

    subprocess = importlib.import_module("subprocess")
    subprocess.Popen = fake_popen

    params = VideoParams(
        width=1,
        height=1,
        frames=1,
        steps=1,
        offload=True,
        t5_cpu=True,
        precision="fp16",
    )
    worker = workers.VideoWorker("hello", "", params)
    worker.progress = DummySignal()
    worker.finished = DummySignal()
    worker.error = DummySignal()
    workers.VideoWorker.run(worker)
    cmd = captured_cmd[0]
    assert cmd[0] == "wan2.2"
    assert "--offload" in cmd
    assert "--t5_cpu" in cmd
    assert worker.progress.emitted == [10, 100]
    assert worker.finished.emitted == [os.path.abspath("output.mp4")]
    assert worker.error.emitted == []


def test_image_worker_stop_prevents_progress():
    fake_model_manager.ModelManager.get_flux_pipeline = (
        lambda params: FakePipeline()
    )
    params = ImageParams(width=1, height=1, steps=2, guidance=1)
    worker = workers.ImageWorker("prompt", "", params)
    worker.progress = DummySignal()
    worker.result = DummySignal()
    worker.error = DummySignal()

    def on_progress(value):
        if value >= 0:
            worker.stop()

    worker.progress.connect(on_progress)
    workers.ImageWorker.run(worker)
    assert worker.progress.emitted == [0, 50]
    assert worker.result.emitted == []
    assert worker.error.emitted == []


def test_video_worker_stop_prevents_progress():
    def fake_popen(cmd, stdout, stderr, text):
        class Proc:
            returncode = 0
            stdout = ["Progress: 10%\n", "Progress: 100%\n"]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def kill(self):
                pass

        return Proc()

    subprocess = importlib.import_module("subprocess")
    subprocess.Popen = fake_popen

    params = VideoParams(
        width=1,
        height=1,
        frames=1,
        steps=1,
        offload=False,
        t5_cpu=False,
        precision="fp16",
    )
    worker = workers.VideoWorker("p", "", params)
    worker.progress = DummySignal()
    worker.finished = DummySignal()
    worker.error = DummySignal()

    def on_progress(value):
        worker.stop()

    worker.progress.connect(on_progress)
    workers.VideoWorker.run(worker)
    assert worker.progress.emitted == [10]
    assert worker.finished.emitted == []
    assert worker.error.emitted == []
