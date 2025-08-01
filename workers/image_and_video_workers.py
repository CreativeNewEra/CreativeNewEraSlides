import logging
import os
import subprocess
from dataclasses import asdict
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage
import torch

from PIL import Image

from .params import ImageParams, VideoParams

logger = logging.getLogger(__name__)


def pil_to_qimage(pil_image: Image.Image) -> QImage:
    """Convert a PIL Image to a :class:`QImage` for Qt display."""
    if pil_image.mode != "RGBA":
        pil_image = pil_image.convert("RGBA")
    data = pil_image.tobytes("raw", "RGBA")
    qimg = QImage(
        data,
        pil_image.width,
        pil_image.height,
        QImage.Format_RGBA8888,
    )
    return qimg


class ImageWorker(QThread):
    """Run Flux image generation in a thread to keep the UI responsive."""

    progress = pyqtSignal(int)  # emits percentage progress
    result = pyqtSignal(QImage)  # emits final image
    error = pyqtSignal(str)  # emits error message

    def __init__(
        self,
        prompt: str,
        neg_prompt: str,
        params: ImageParams,
        parent: Optional[QObject] = None,
    ) -> None:
        """Initialize the worker.

        Parameters:
            prompt: Text prompt for the model.
            neg_prompt: Negative prompt used to avoid undesired content.
            params: Image generation parameters.
            parent: Optional QObject to set as the thread parent.
        """
        super().__init__(parent)
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.params = params
        self._running = True

    def run(self) -> None:
        """Execute image generation and emit progress and result signals."""
        try:
            # Load or reuse cached pipeline
            from utils.model_manager import (
                ModelManager,
            )  # Ensure ModelManager exists in this module

            pipe = ModelManager.get_flux_pipeline(asdict(self.params))
            if self.params.quantized:
                logger.info("Using quantized weights for image generation")

            # Generate image with progress callback
            total_steps = self.params.steps
            self.progress.emit(0)

            def _callback(step, timestep, latents):
                if not self._running:
                    return
                if total_steps > 0:
                    pct = min(100, int((step + 1) / total_steps * 100))
                else:
                    pct = 0  # Default to 0% if total_steps is invalid
                self.progress.emit(pct)

            out = pipe(
                prompt=self.prompt,
                negative_prompt=self.neg_prompt or None,
                width=self.params.width,
                height=self.params.height,
                num_inference_steps=total_steps,
                guidance_scale=self.params.guidance,
                callback=_callback,
                callback_steps=1,
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
            logger.exception("Image generation failed: %s", msg)
            self.error.emit(msg)
        finally:
            # Ensure GPU memory is freed
            try:
                torch.cuda.empty_cache()
            except (AttributeError, RuntimeError) as exc:
                from utils.errors import parse_error

                msg = parse_error(exc)
                logger.warning(msg)

    def stop(self) -> None:
        """Signal the thread to stop early."""
        self._running = False


class VideoWorker(QThread):
    """Run Wan2.2 video generation via CLI in a background thread."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(str)  # emits output file path
    error = pyqtSignal(str)

    def __init__(
        self,
        prompt: str,
        neg_prompt: str,
        params: VideoParams,
        parent: Optional[QObject] = None,
    ) -> None:
        """Initialize the worker.

        Parameters:
            prompt: Text prompt for the model.
            neg_prompt: Negative prompt used to avoid undesired content.
            params: Video generation parameters.
            parent: Optional QObject to set as the thread parent.
        """
        super().__init__(parent)
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.params = params
        self._running = True

    def run(self) -> None:
        """Execute video generation using Wan2.2 and emit signals."""
        try:
            from utils.model_manager import ModelManager

            # Check if Wan2.2 model exists
            wan_model_path = ModelManager.get_wan_model_path()
            if not os.path.exists(wan_model_path):
                raise FileNotFoundError(f"Wan2.2 model not found at {wan_model_path}")

            # Look for inference script in the model directory
            inference_script = None
            possible_scripts = [
                os.path.join(wan_model_path, "inference.py"),
                os.path.join(wan_model_path, "sample.py"),
                os.path.join(wan_model_path, "generate.py"),
                os.path.join(wan_model_path, "run_inference.py"),
            ]

            for script in possible_scripts:
                if os.path.exists(script):
                    inference_script = script
                    break

            if not inference_script:
                raise FileNotFoundError(
                    f"No inference script found in {wan_model_path}"
                )

            # Build command to run the Python script
            cmd = [
                "python",
                inference_script,
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
            if self.params.offload:
                cmd.append("--offload_model")  # Common flag name
            if self.params.t5_cpu:
                cmd.append("--t5_cpu")
            cmd += ["--convert_model_dtype", self.params.precision]

            logger.info(f"Running Wan2.2 inference: {' '.join(cmd)}")

            # Launch process and ensure it closes properly
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=wan_model_path,  # Run from model directory
            ) as proc:
                # Parse stdout for progress updates
                for line in proc.stdout:
                    if not self._running:
                        proc.kill()
                        return
                    # expect lines like "Progress: 42%"
                    if "Progress:" in line:
                        try:
                            progress_part = line.split("Progress:")[1]
                            pct = int(progress_part.strip().rstrip("%"))
                            self.progress.emit(pct)
                        except ValueError:
                            logger.warning(
                                "Unexpected progress line: %s",
                                line.strip(),
                            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Wan2.2 failed with code {proc.returncode}",
                )

            # Assuming output.mp4 in working dir
            out_file = os.path.abspath("output.mp4")
            self.finished.emit(out_file)
        except Exception as e:
            from utils.errors import parse_error

            msg = parse_error(e)
            logger.exception("Video generation failed: %s", msg)
            self.error.emit(msg)

    def stop(self) -> None:
        """Signal the thread to stop early."""
        self._running = False
