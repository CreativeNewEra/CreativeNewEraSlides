#!/usr/bin/env python3
"""Setup script to download all required AI models."""

import sys
import logging
from utils.model_downloader import ModelDownloader


def main():
    """Download all required AI models."""
    print("🚀 CreativeNewEraSlides Model Setup")
    print("=" * 50)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Get Hugging Face token
    token = input("Enter your Hugging Face token: ").strip()
    if not token:
        print("❌ Error: Hugging Face token is required!")
        sys.exit(1)

    # Initialize downloader
    downloader = ModelDownloader()

    # Show what will be downloaded
    sizes = downloader.get_download_size_estimate()
    print("\n📦 Models to download:")
    for model, size in sizes.items():
        exists = (
            "✅ Already exists"
            if downloader.check_model_exists(model)
            else "⬇️  Will download"
        )
        print(f"  {model}: {size} - {exists}")

    # Confirm download
    if input("\nProceed with download? (y/N): ").lower() != "y":
        print("❌ Setup cancelled.")
        sys.exit(0)

    print("\n🔐 Authenticating with Hugging Face...")
    if not downloader.authenticate_huggingface(token):
        print("❌ Authentication failed!")
        sys.exit(1)

    print("✅ Authentication successful!")

    # Download models
    print("\n⬇️  Starting model downloads...")
    results = downloader.download_all_models()

    # Show results
    print("\n📊 Download Results:")
    all_success = True
    for model, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {model}: {status}")
        if not success:
            all_success = False

    if all_success:
        print("\n🎉 All models downloaded successfully!")
        print("You can now run the application with: ./run.sh")
    else:
        print("\n⚠️  Some models failed to download.")
        print("Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
