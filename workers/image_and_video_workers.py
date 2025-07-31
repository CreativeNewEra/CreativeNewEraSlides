import os
import subprocess
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import torch

from PIL import Image


def pil_to_qimage(pil_image: Image.Image) -> QImage:
    """
    Convert PIL Image to QImage for Qt display.
    """
    if pil_image.mode != "RGBA":
        pil_image = pil_image.convert("RGBA")
    data = pil_image.tobytes("raw", "RGBA")
    qimg = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
    return qimg


class ImageWorker(QThread):
    """
    QThread worker to run Flux image generation without blocking UI.
    """
    progress = pyqtSignal(int)          # emits percentage progress
    result = pyqtSignal(QImage)         # emits final image
    error  = pyqtSignal(str)            # emits error message

    def __init__(self, prompt: str, neg_prompt: str, params: dict, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.params = params
        self._running = True

    def run(self):
        try:
            # Load or reuse cached pipeline
            from utils.model_manager import ModelManager  # Ensure ModelManager exists in this module
            pipe = ModelManager.get_flux_pipeline(self.params)

            # Generate image
            out = pipe(
                prompt=self.prompt,
                negative_prompt=self.neg_prompt or None,
                width=self.params['width'],
                height=self.params['height'],
                num_inference_steps=self.params['steps'],
                guidance_scale=self.params['guidance'],
            )
            if not self._running:
                return
            pil_img = out.images[0]
            qimg = pil_to_qimage(pil_img)
            self.result.emit(qimg)
        except Exception as e:
            # Parse and emit user-friendly error
            from utils.errors import parse_error
            msg = parse_error(e)
            self.error.emit(msg)
        finally:
            # Ensure GPU memory is freed
            try:
                torch.cuda.empty_cache()
            except:
                pass

    def stop(self):
        """
        Signal the thread to stop early.
        """
        self._running = False


class VideoWorker(QThread):
    """
    QThread worker to run Wan2.2 video generation via CLI.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)          # emits output file path
    error    = pyqtSignal(str)

    def __init__(self, prompt: str, neg_prompt: str, params: dict, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.params = params
        self._running = True

    def run(self):
        try:
            # Build CLI command
            cmd = [
                'wan2.2',
                '--prompt', self.prompt,
                '--width', str(self.params['width']),
                '--height', str(self.params['height']),
                '--frames', str(self.params['frames']),
                '--steps', str(self.params['steps']),
            ]
            if self.neg_prompt:
                cmd += ['--neg_prompt', self.neg_prompt]
            if self.params.get('offload'):
                cmd.append('--offload')
            if self.params.get('t5_cpu'):
                cmd.append('--t5_cpu')
            cmd += ['--precision', self.params.get('precision', 'fp16')]

            # Launch process
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            # Parse stdout for progress updates
            for line in proc.stdout:
                if not self._running:
                    proc.kill()
                    return
                # expect lines like "Progress: 42%"
                if 'Progress:' in line:
                    try:
                        pct = int(line.split('Progress:')[1].strip().rstrip('%'))
                        self.progress.emit(pct)
                    except:
                        pass
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Wan2.2 failed with code {proc.returncode}")

            # Assuming output.mp4 in working dir
            out_file = os.path.abspath('output.mp4')
            self.finished.emit(out_file)
        except Exception as e:
            from utils.errors import parse_error
            msg = parse_error(e)
            self.error.emit(msg)

    def stop(self):
        self._running = False
