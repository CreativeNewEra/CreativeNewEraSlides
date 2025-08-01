"""Automatic model downloader for Flux and Wan2.2 models from Hugging Face."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from huggingface_hub import hf_hub_download, login, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import torch

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Handles automatic downloading of AI models from Hugging Face."""
    
    # Model configurations
    MODELS_CONFIG = {
        "flux": {
            "repo_id": "black-forest-labs/FLUX.1-dev",
            "files": [
                "flux1-dev.safetensors",
                "ae.safetensors", 
                "text_encoder/model.safetensors",
                "text_encoder_2/model.safetensors",
                "tokenizer/tokenizer.json",
                "tokenizer_2/tokenizer.json",
                "scheduler/scheduler_config.json"
            ],
            "local_dir": "Models/Flux",
            "requires_auth": True
        },
        "wan2.2": {
            "repo_id": "Wan-AI/Wan2.2-TI2V-5B",
            "files": "all",  # Download entire repository
            "local_dir": "Models/Wan2.2",
            "requires_auth": False
        }
    }
    
    def __init__(self, base_path: str = "."):
        """Initialize model downloader.
        
        Args:
            base_path: Base directory for the project
        """
        self.base_path = Path(base_path)
        self.models_dir = self.base_path / "Models"
        self.models_dir.mkdir(exist_ok=True)
        
    def authenticate_huggingface(self, token: str) -> bool:
        """Authenticate with Hugging Face using token.
        
        Args:
            token: Hugging Face authentication token
            
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            login(token=token)
            logger.info("Successfully authenticated with Hugging Face")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with Hugging Face: {e}")
            return False
    
    def check_model_exists(self, model_name: str) -> bool:
        """Check if model files already exist locally.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model exists, False otherwise
        """
        if model_name not in self.MODELS_CONFIG:
            return False
            
        config = self.MODELS_CONFIG[model_name]
        local_dir = self.models_dir / config["local_dir"].split("/")[-1]
        
        if not local_dir.exists():
            return False
            
        # Check if directory has files
        files = list(local_dir.rglob("*"))
        if files:
            # Special handling for Flux - if we only have a single .safetensors file,
            # we can consider it "exists" but incomplete
            if model_name == "flux":
                safetensors_files = list(local_dir.glob("*.safetensors"))
                if len(safetensors_files) == 1 and len(files) == 1:
                    logger.info(f"Model {model_name} has single file at {local_dir}, may need full pipeline")
                    return True  # Exists but may be incomplete
                elif (local_dir / "model_index.json").exists():
                    logger.info(f"Model {model_name} complete pipeline exists at {local_dir}")
                    return True
                else:
                    logger.info(f"Model {model_name} partial files at {local_dir}")
                    return True
            else:
                logger.info(f"Model {model_name} already exists at {local_dir}")
                return True
            
        return False
    
    def download_model(self, model_name: str, force_download: bool = False) -> bool:
        """Download a specific model from Hugging Face.
        
        Args:
            model_name: Name of the model to download
            force_download: Whether to re-download even if model exists
            
        Returns:
            True if download successful, False otherwise
        """
        if model_name not in self.MODELS_CONFIG:
            logger.error(f"Unknown model: {model_name}")
            return False
            
        config = self.MODELS_CONFIG[model_name]
        local_dir = self.models_dir / config["local_dir"].split("/")[-1]
        
        # Check if already exists
        if not force_download and self.check_model_exists(model_name):
            return True
            
        try:
            logger.info(f"Downloading {model_name} from {config['repo_id']}...")
            local_dir.mkdir(parents=True, exist_ok=True)
            
            if config["files"] == "all":
                # Download entire repository
                snapshot_download(
                    repo_id=config["repo_id"],
                    local_dir=str(local_dir),
                    token=True if config["requires_auth"] else None,
                    resume_download=True
                )
            else:
                # Download specific files
                for file_path in config["files"]:
                    try:
                        hf_hub_download(
                            repo_id=config["repo_id"],
                            filename=file_path,
                            local_dir=str(local_dir),
                            token=True if config["requires_auth"] else None,
                            resume_download=True
                        )
                        logger.info(f"Downloaded {file_path}")
                    except HfHubHTTPError as e:
                        if "404" in str(e):
                            logger.warning(f"File not found, skipping: {file_path}")
                            continue
                        else:
                            raise
                            
            logger.info(f"Successfully downloaded {model_name} to {local_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False
    
    def download_all_models(self, force_download: bool = False) -> Dict[str, bool]:
        """Download all configured models.
        
        Args:
            force_download: Whether to re-download existing models
            
        Returns:
            Dictionary mapping model names to download success status
        """
        results = {}
        
        for model_name in self.MODELS_CONFIG.keys():
            logger.info(f"Processing {model_name}...")
            results[model_name] = self.download_model(model_name, force_download)
            
        return results
    
    def get_download_size_estimate(self) -> Dict[str, str]:
        """Get estimated download sizes for models.
        
        Returns:
            Dictionary mapping model names to size estimates
        """
        return {
            "flux": "~23GB (quantized version available)",
            "wan2.2": "~5GB (TI2V-5B variant)"
        }
    
    def setup_models_with_token(self, token: str, force_download: bool = False) -> bool:
        """Complete model setup with authentication.
        
        Args:
            token: Hugging Face authentication token
            force_download: Whether to re-download existing models
            
        Returns:
            True if all models downloaded successfully
        """
        # Authenticate
        if not self.authenticate_huggingface(token):
            return False
            
        # Show download estimates
        sizes = self.get_download_size_estimate()
        logger.info("Download size estimates:")
        for model, size in sizes.items():
            logger.info(f"  {model}: {size}")
            
        # Download all models
        results = self.download_all_models(force_download)
        
        # Check results
        success = all(results.values())
        if success:
            logger.info("All models downloaded successfully!")
        else:
            failed = [name for name, status in results.items() if not status]
            logger.error(f"Failed to download: {failed}")
            
        return success


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download AI models from Hugging Face")
    parser.add_argument("--token", required=True, help="Hugging Face authentication token")
    parser.add_argument("--force", action="store_true", help="Force re-download existing models")
    parser.add_argument("--model", help="Download specific model only")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    downloader = ModelDownloader()
    
    if args.model:
        success = downloader.setup_models_with_token(args.token)
        if success:
            success = downloader.download_model(args.model, args.force)
    else:
        success = downloader.setup_models_with_token(args.token, args.force)
    
    if success:
        print("Model download completed successfully!")
    else:
        print("Model download failed!")
        exit(1)


if __name__ == "__main__":
    main()