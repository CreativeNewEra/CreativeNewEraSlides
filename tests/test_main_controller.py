import importlib
import pathlib
import sys
import types
import builtins

# ---- Stub PyQt5 modules with QTest ----


class DummySignal:
    def __init__(self):
        self.emitted = []
        self._func = None

    def connect(self, func):
        self._func = func

    def emit(self, value=None):
        self.emitted.append(value)
        if self._func:
            if value is None:
                self._func()
            else:
                self._func(value)


class QApplication:
    def __init__(self, *args, **kwargs):
        pass

    def exec_(self):
        return 0


class DummyStatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, msg):
        self.messages.append(msg)


class QMainWindow:
    def __init__(self):
        self._status = DummyStatusBar()

    def statusBar(self):
        return self._status

    def show(self):
        pass


class QPushButton:
    def __init__(self):
        self.clicked = DummySignal()

    def click(self):
        self.clicked.emit()


class QComboBox:
    def __init__(self):
        self.items = []
        self.current = None

    def clear(self):
        self.items = []

    def addItems(self, items):
        self.items.extend(items)

    def setCurrentText(self, text):
        self.current = text

    def currentText(self):
        if self.current is not None:
            return self.current
        return self.items[0] if self.items else ""


class QTextEdit:
    def __init__(self):
        self.text = ""
        self.history = []

    def toPlainText(self):
        return self.text

    def trackHistory(self):
        self.history.append(self.text)
class QSpinBox:
    def __init__(self, value=0):
        self._value = value

    def value(self):
        return self._value


class QCheckBox:
    def __init__(self, checked=False):
        self._checked = checked

    def isChecked(self):
        return self._checked


class QProgressBar:
    def __init__(self):
        self.values = []

    def setValue(self, v):
        self.values.append(v)


class QLabel:
    pass


class QTest:
    @staticmethod
    def mouseClick(widget, button):
        widget.click()


class Qt:
    LeftButton = 1
    KeepAspectRatio = 1
    SmoothTransformation = 1


class QPixmap:
    @staticmethod
    def fromImage(img):
        return QPixmap()

    def scaled(self, *args, **kwargs):
        return self


class QImage:
    pass


class QCloseEvent:
    pass

sys.modules["PyQt5.QtGui.QCloseEvent"] = QCloseEvent
pyqt5 = types.ModuleType("PyQt5")
qtwidgets = types.ModuleType("PyQt5.QtWidgets")
qtcore = types.ModuleType("PyQt5.QtCore")
qtgui = types.ModuleType("PyQt5.QtGui")
qttest = types.ModuleType("PyQt5.QtTest")

qtwidgets.QApplication = QApplication
qtwidgets.QMainWindow = QMainWindow
qtwidgets.QPushButton = QPushButton
qtwidgets.QComboBox = QComboBox
qtwidgets.QTextEdit = QTextEdit
qtwidgets.QSpinBox = QSpinBox
qtwidgets.QCheckBox = QCheckBox
qtwidgets.QProgressBar = QProgressBar
qtwidgets.QLabel = QLabel

qtcore.Qt = Qt
qtcore.QCloseEvent = QCloseEvent
qtcore.QSettings = type(
    "QSettings",
    (),
    {
        "__init__": lambda self, *a, **kw: setattr(self, "_s", {}),
        "value": lambda self, k, d=None: self._s.get(k, d),
        "setValue": lambda self, k, v: self._s.__setitem__(k, v),
    },
)

qtgui.QPixmap = QPixmap
qtgui.QImage = QImage

qttest.QTest = QTest

sys.modules["PyQt5"] = pyqt5
sys.modules["PyQt5.QtWidgets"] = qtwidgets
sys.modules["PyQt5.QtCore"] = qtcore
sys.modules["PyQt5.QtGui"] = qtgui
sys.modules["PyQt5.QtTest"] = qttest

# ---- Stub heavy dependencies ----


class FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def empty_cache():
        pass


fake_torch = types.SimpleNamespace(
    float16="float16",
    float32="float32",
    cuda=FakeCuda,
)

sys.modules["torch"] = fake_torch
sys.modules["diffusers"] = types.ModuleType("diffusers")

# ---- Stub ui.main_window ----


class DummyUI:
    def setupUi(self, window):
        self.prompt_edit = QTextEdit()
        self.neg_prompt_edit = QTextEdit()
        self.width_spin = QSpinBox(512)
        self.height_spin = QSpinBox(512)
        self.steps_spin = QSpinBox(10)
        self.guidance_spin = QSpinBox(7)
        self.device_combo = QComboBox()
        self.quant_checkbox = QCheckBox(False)
        self.gen_button = QPushButton()
        self.image_progress = QProgressBar()
        self.image_display = QLabel()
        self.status_bar = window.statusBar()

        self.video_prompt_edit = QTextEdit()
        self.video_neg_prompt_edit = QTextEdit()
        self.video_width_spin = QSpinBox(256)
        self.video_height_spin = QSpinBox(256)
        self.frames_spin = QSpinBox(1)
        self.video_steps_spin = QSpinBox(10)
        self.offload_checkbox = QCheckBox(False)
        self.t5_cpu_checkbox = QCheckBox(False)
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["fp16"])
        self.video_button = QPushButton()
        self.video_progress = QProgressBar()


ui_module = types.ModuleType("ui.main_window")
ui_module.Ui_MainWindow = DummyUI
sys.modules["ui.main_window"] = ui_module

# ---- Stub workers ----


class DummyImageWorker:
    def __init__(self, prompt, neg_prompt, params, parent=None):
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.params = params
        self.started = False
        self._running = True
        self.progress = DummySignal()
        self.result = DummySignal()
        self.error = DummySignal()

    def start(self):
        self.started = True

    def run(self):
        try:
            from dataclasses import asdict
            from utils.model_manager import ModelManager
            import torch
            pipe = ModelManager.get_flux_pipeline(asdict(self.params))
            self.progress.emit(0)
            total = self.params.steps
            for i in range(total):
                pct = min(100, int((i + 1) / total * 100))
                self.progress.emit(pct)
                if not self._running:
                    break
            if self._running:
                pipe()
                self.result.emit(object())
        except Exception as e:
            self.error.emit(f"Runtime error: {e}")
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def stop(self):
        self._running = False

    def isRunning(self):
        return self._running

    def wait(self):
        pass


class DummyVideoWorker:
    def __init__(self, prompt, neg_prompt, params, parent=None):
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.params = params
        self.started = False
        self._running = True
        self.progress = DummySignal()
        self.finished = DummySignal()
        self.error = DummySignal()

    def start(self):
        self.started = True

    def run(self):
        import os
        import subprocess

        cmd = [
            DEFAULT_COMMAND,
            "--prompt",
            self.prompt,
            "--width",
            str(self.params.width),
            "--height",
            str(self.params.height),
            "--frames",
            str(self.params.frames),
            "--steps",
            str(self.params.steps),
        ]
        if self.neg_prompt:
            cmd += ["--neg_prompt", self.neg_prompt]
        if getattr(self.params, "offload", False):
            cmd.append("--offload")
        if getattr(self.params, "t5_cpu", False):
            cmd.append("--t5_cpu")
        cmd += ["--precision", self.params.precision]

        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ) as proc:
                for line in proc.stdout:
                    if not self._running:
                        proc.kill()
                        return
                    if "Progress:" in line:
                        pct = int(line.split("Progress:")[1].strip().rstrip("%"))
                        self.progress.emit(pct)
            if self._running:
                if proc.returncode != 0:
                    raise RuntimeError(f"Wan2.2 failed with code {proc.returncode}")
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                    out_file = temp_file.name
                self.finished.emit(out_file)
        except Exception as e:
            self.error.emit(f"Runtime error: {e}")

    def stop(self):
        self._running = False

    def isRunning(self):
        return self._running

    def wait(self):
        pass


workers_module = types.ModuleType("workers.image_and_video_workers")
workers_module.ImageWorker = DummyImageWorker
workers_module.VideoWorker = DummyVideoWorker
sys.modules["workers.image_and_video_workers"] = workers_module

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

main_controller = importlib.import_module("controllers.main_controller")
sys.modules.pop("diffusers", None)


class DummySettings:
    def __init__(self):
        self.store = {}

    def get(self, key, default=None):
        return self.store.get(key, default)

    def set(self, key, value):
        self.store[key] = value

    def get_model_path(self, key, default=""):
        return default


main_controller.SettingsManager = DummySettings


def test_device_list_populated():
    controller = main_controller.MainController()
    assert controller.ui.device_combo.items == ["cpu", "cuda:0"]
    assert controller.ui.device_combo.current == "cpu"


def test_workers_start_and_prompt_history():
    controller = main_controller.MainController()
    controller.ui.prompt_edit.text = "hello"
    controller.ui.video_prompt_edit.text = "world"
    QTest.mouseClick(controller.ui.gen_button, Qt.LeftButton)
    assert isinstance(controller.image_worker, DummyImageWorker)
    assert controller.image_worker.started
    QTest.mouseClick(controller.ui.video_button, Qt.LeftButton)
    assert isinstance(controller.video_worker, DummyVideoWorker)
    assert controller.video_worker.started
    assert controller.ui.prompt_edit.history == ["hello"]
    assert controller.ui.video_prompt_edit.history == ["world"]
