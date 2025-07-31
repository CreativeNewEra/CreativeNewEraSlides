import sys
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from ui.main_window import Ui_MainWindow
from utils.settings_manager import SettingsManager
from workers.image_and_video_workers import ImageWorker, VideoWorker


class MainController:
    """
    Orchestrates the UI, settings, and worker threads.
    """
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.window)
        self.settings = SettingsManager()

        # Populate devices and bind actions
        self._populate_device_list()
        self._bind_signals()

        # Show window; event loop is started via run()
        self.window.show()

    def run(self):
        """Start the Qt event loop and return the exit code."""
        return self.app.exec_()

    def _populate_device_list(self):
        # Detect available devices
        devices = ["cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        # Update combo box
        self.ui.device_combo.clear()
        self.ui.device_combo.addItems(devices)
        # Restore last used device
        last = self.settings.get("device", "cpu")
        if last in devices:
            self.ui.device_combo.setCurrentText(last)

    def _bind_signals(self):
        # Image generation
        self.ui.gen_button.clicked.connect(self.start_image_generation)
        # Video generation
        self.ui.video_button.clicked.connect(self.start_video_generation)

    def start_image_generation(self):
        prompt = self.ui.prompt_edit.toPlainText().strip()
        neg    = self.ui.neg_prompt_edit.toPlainText().strip()
        params = {
            'width': self.ui.width_spin.value(),
            'height': self.ui.height_spin.value(),
            'steps': self.ui.steps_spin.value(),
            'guidance': self.ui.guidance_spin.value(),
            'model_path': self.settings.get_model_path('flux'),
            'device': self.ui.device_combo.currentText(),
        }
        # Persist chosen device
        self.settings.set("device", params['device'])

        # Start worker
        self.image_worker = ImageWorker(prompt, neg, params)
        self.image_worker.progress.connect(self.ui.image_progress.setValue)
        self.image_worker.result.connect(self._on_image_result)
        self.image_worker.error.connect(self._handle_error)
        self.image_worker.start()

        self.ui.status_bar.showMessage("Generating image...")

    def _on_image_result(self, qimg):
        pixmap = QPixmap.fromImage(qimg)
        self.ui.image_display.setPixmap(
            pixmap.scaled(
                self.ui.image_display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
        self.ui.status_bar.showMessage("Image generation complete")

    def start_video_generation(self):
        prompt = self.ui.video_prompt_edit.toPlainText().strip()
        neg    = self.ui.video_neg_prompt_edit.toPlainText().strip()
        params = {
            'width': self.ui.video_width_spin.value(),
            'height': self.ui.video_height_spin.value(),
            'frames': self.ui.frames_spin.value(),
            'steps': self.ui.video_steps_spin.value(),
            'offload': self.ui.offload_checkbox.isChecked(),
            't5_cpu': self.ui.t5_cpu_checkbox.isChecked(),
            'precision': self.ui.precision_combo.currentText(),
        }

        self.video_worker = VideoWorker(prompt, neg, params)
        self.video_worker.progress.connect(self.ui.video_progress.setValue)
        self.video_worker.finished.connect(self._on_video_finished)
        self.video_worker.error.connect(self._handle_error)
        self.video_worker.start()

        self.ui.status_bar.showMessage("Generating video...")

    def _on_video_finished(self, path: str):
        self.ui.status_bar.showMessage(f"Video saved to {path}")

    def _handle_error(self, msg: str):
        # Display errors in status bar (could be extended to dialogs)
        self.ui.status_bar.showMessage(msg)


if __name__ == '__main__':
    MainController().run()
