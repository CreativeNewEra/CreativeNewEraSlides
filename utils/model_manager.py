from diffusers import FluxPipeline, StableDiffusionPipeline
import logging
import threading
import torch
from pathlib import Path

from .settings_manager import SettingsManager
from .model_downloader import ModelDownloader


logger = logging.getLogger(__name__)


class ModelManager:
    """Caches and returns loaded models/pipelines."""

    _flux_pipe = None
    _flux_device = None
    _settings_manager = None
    _model_downloader = None
    _flux_lock = threading.Lock()

    @classmethod
    def _get_settings_manager(cls):
        if cls._settings_manager is None:
            cls._settings_manager = SettingsManager()
        return cls._settings_manager
    
    @classmethod
    def _get_model_downloader(cls):
        if cls._model_downloader is None:
            cls._model_downloader = ModelDownloader()
        return cls._model_downloader
    
    @classmethod
    def _ensure_models_available(cls):
        """Ensure required models are downloaded."""
        downloader = cls._get_model_downloader()
        
        # Check if models exist
        flux_exists = downloader.check_model_exists("flux")
        wan_exists = downloader.check_model_exists("wan2.2")
        
        if not flux_exists or not wan_exists:
            missing = []
            if not flux_exists:
                missing.append("Flux")
            if not wan_exists:
                missing.append("Wan2.2")
            
            raise RuntimeError(
                f"Missing models: {', '.join(missing)}. "
                f"Please run 'python setup_models.py' to download them."
            )

    @classmethod
    def _load_flux_from_single_file(cls, model_path: str, dtype, device: str):
        """Load Flux pipeline from a single .safetensors file."""
        from transformers import CLIPTextModel, T5EncoderModel
        from diffusers import AutoencoderKL
        
        # Check if model_path is a directory with a single .safetensors file
        model_path_obj = Path(model_path)
        if model_path_obj.is_dir():
            safetensors_files = list(model_path_obj.glob("*.safetensors"))
            if len(safetensors_files) == 1:
                model_file = safetensors_files[0]
                logger.info(f"Found single model file: {model_file}")
                
                # For Flux models that are single files, we need to load the components separately
                # and then combine them into a pipeline
                
                # Try FluxPipeline first with automatic component loading
                for pipeline_class in [FluxPipeline, StableDiffusionPipeline]:
                    try:
                        logger.info(f"Trying {pipeline_class.__name__}.from_single_file with auto components...")
                        
                        # Load with component auto-loading (this will download missing components)
                        pipe = pipeline_class.from_single_file(
                            str(model_file),
                            torch_dtype=dtype,
                        )
                        logger.info(f"Successfully loaded {pipeline_class.__name__} from single file")
                        return pipe
                        
                    except Exception as e:
                        logger.warning(f"{pipeline_class.__name__}.from_single_file failed: {e}")
                        
                        # If that fails, try with explicit text encoder loading
                        if "text_encoder" in str(e).lower() or "clip" in str(e).lower():
                            try:
                                logger.info(f"Trying {pipeline_class.__name__} with explicit component loading...")
                                
                                # Load default text encoder for the pipeline
                                if pipeline_class == FluxPipeline:
                                    # For Flux, we need both CLIP and T5 text encoders
                                    text_encoder = CLIPTextModel.from_pretrained(
                                        "openai/clip-vit-large-patch14", 
                                        torch_dtype=dtype
                                    )
                                    text_encoder_2 = T5EncoderModel.from_pretrained(
                                        "google/t5-v1_1-xxl",
                                        torch_dtype=dtype
                                    )
                                    pipe = pipeline_class.from_single_file(
                                        str(model_file),
                                        text_encoder=text_encoder,
                                        text_encoder_2=text_encoder_2,
                                        torch_dtype=dtype,
                                    )
                                else:
                                    # For SD, just CLIP
                                    text_encoder = CLIPTextModel.from_pretrained(
                                        "openai/clip-vit-large-patch14",
                                        torch_dtype=dtype
                                    )
                                    pipe = pipeline_class.from_single_file(
                                        str(model_file),
                                        text_encoder=text_encoder,
                                        torch_dtype=dtype,
                                    )
                                    
                                logger.info(f"Successfully loaded {pipeline_class.__name__} with explicit components")
                                return pipe
                                
                            except Exception as e2:
                                logger.warning(f"Explicit component loading also failed: {e2}")
                                continue
                        else:
                            continue
                
                # If all methods fail, raise error
                raise RuntimeError(f"Could not load model file {model_file} with any method")
        
        raise ValueError(f"Could not find single safetensors file in {model_path}")

    @classmethod
    def get_flux_pipeline(cls, params: dict):
        # Skip model availability check for now since we have a single file
        # cls._ensure_models_available()
        
        settings_manager = cls._get_settings_manager()
        # Use downloaded model path or existing custom path
        default_model_path = str(Path("Models/Flux").absolute())
        model_path = params.get('model_path') or settings_manager.get_model_path('flux') or default_model_path
        requested_device = params.get('device') or settings_manager.get_device()
        dtype = torch.bfloat16 if requested_device != "cpu" else torch.float32

        with cls._flux_lock:
            if cls._flux_pipe is None:
                logger.info(f"Loading Flux pipeline from {model_path}")
                
                # Check if it's a directory with pipeline structure
                model_path_obj = Path(model_path)
                has_model_index = (model_path_obj / "model_index.json").exists()
                
                if has_model_index:
                    # Standard pipeline loading
                    try:
                        pipe = FluxPipeline.from_pretrained(
                            model_path,
                            torch_dtype=dtype,
                        )
                    except Exception as e:
                        logger.warning(f"FluxPipeline failed, trying StableDiffusionPipeline: {e}")
                        pipe = StableDiffusionPipeline.from_pretrained(
                            model_path,
                            torch_dtype=dtype,
                        )
                else:
                    # Single file loading
                    pipe = cls._load_flux_from_single_file(model_path, dtype, requested_device)
                
                # Apply memory optimizations for 16GB VRAM
                if requested_device != "cpu":
                    pipe.enable_model_cpu_offload()
                    if hasattr(pipe.vae, 'enable_slicing'):
                        pipe.vae.enable_slicing()
                    if hasattr(pipe.vae, 'enable_tiling'):
                        pipe.vae.enable_tiling()
                
                pipe.to(requested_device)
                cls._flux_pipe = pipe
                cls._flux_device = requested_device
                logger.info("Flux pipeline loaded successfully")
            elif cls._flux_device != requested_device:
                pipe = cls._flux_pipe
                try:
                    pipe.to(requested_device)
                except (RuntimeError, OSError):
                    logger.exception(
                        "Failed moving flux pipeline to %s; reloading", requested_device
                    )
                    try:
                        pipe = FluxPipeline.from_pretrained(
                            model_path,
                            torch_dtype=dtype,
                        )
                        if requested_device != "cpu":
                            pipe.enable_model_cpu_offload()
                            if hasattr(pipe.vae, 'enable_slicing'):
                                pipe.vae.enable_slicing()
                            if hasattr(pipe.vae, 'enable_tiling'):
                                pipe.vae.enable_tiling()
                        pipe.to(requested_device)
                        cls._flux_pipe = pipe
                    except (RuntimeError, OSError) as load_exc:
                        raise RuntimeError(
                            f"Could not load flux pipeline on {requested_device}"
                        ) from load_exc
                cls._flux_device = requested_device

            return cls._flux_pipe
    
    @classmethod
    def get_wan_model_path(cls):
        """Get the path to the Wan2.2 model."""
        cls._ensure_models_available()
        return str(Path("Models/Wan2.2").absolute())
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached models and free memory."""
        with cls._flux_lock:
            if cls._flux_pipe is not None:
                del cls._flux_pipe
                cls._flux_pipe = None
                cls._flux_device = None
                
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Model cache cleared")
