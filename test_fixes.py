#!/usr/bin/env python3
"""Test script to verify the fixes work."""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_model_detection():
    """Test model detection and loading logic."""
    print("🔍 Testing model detection...")
    
    from utils.model_downloader import ModelDownloader
    downloader = ModelDownloader()
    
    flux_exists = downloader.check_model_exists('flux')
    wan_exists = downloader.check_model_exists('wan2.2')
    
    print(f"   Flux model exists: {'✅' if flux_exists else '❌'}")
    print(f"   Wan2.2 model exists: {'✅' if wan_exists else '❌'}")
    
    return flux_exists, wan_exists

def test_flux_loading_logic():
    """Test Flux model loading logic without actually loading."""
    print("🧪 Testing Flux loading logic...")
    
    from utils.model_manager import ModelManager
    from pathlib import Path
    
    model_path = Path("Models/Flux")
    has_model_index = (model_path / "model_index.json").exists()
    safetensors_files = list(model_path.glob("*.safetensors"))
    
    print(f"   Has model_index.json: {'✅' if has_model_index else '❌'}")
    print(f"   Safetensors files: {len(safetensors_files)}")
    
    if safetensors_files:
        print(f"   Single file detected: {safetensors_files[0].name}")
        print("   ✅ Will use single-file loading method")
    elif has_model_index:
        print("   ✅ Will use standard pipeline loading")
    else:
        print("   ❌ No valid model format detected")
        return False
    
    return True

def test_wan_inference_detection():
    """Test Wan2.2 inference script detection."""
    print("🎬 Testing Wan2.2 inference detection...")
    
    wan_model_path = Path("Models/Wan2.2")
    if not wan_model_path.exists():
        print("   ❌ Wan2.2 directory not found")
        return False
    
    possible_scripts = [
        "inference.py",
        "sample.py",  
        "generate.py",
        "run_inference.py"
    ]
    
    found_scripts = []
    for script in possible_scripts:
        script_path = wan_model_path / script
        if script_path.exists():
            found_scripts.append(script)
    
    if found_scripts:
        print(f"   ✅ Found inference scripts: {found_scripts}")
        return True
    else:
        print(f"   ⚠️  No inference scripts found (this is expected if Wan2.2 not downloaded)")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing CreativeNewEraSlides fixes...")
    print("=" * 50)
    
    # Test 1: Model detection
    flux_exists, wan_exists = test_model_detection()
    
    # Test 2: Flux loading logic
    flux_loading_ok = test_flux_loading_logic()
    
    # Test 3: Wan2.2 inference detection
    wan_inference_ok = test_wan_inference_detection()
    
    print("\n📊 Test Results:")
    print(f"   Model detection: {'✅' if flux_exists else '❌'}")
    print(f"   Flux loading logic: {'✅' if flux_loading_ok else '❌'}")
    print(f"   Wan2.2 inference detection: {'✅' if wan_inference_ok else '⚠️  Expected if not downloaded'}")
    
    if flux_exists and flux_loading_ok:
        print("\n🎉 Flux model should work!")
    else:
        print("\n⚠️  Flux model may have issues")
    
    if wan_exists and wan_inference_ok:
        print("🎉 Wan2.2 video generation should work!")
    elif wan_exists and not wan_inference_ok:
        print("⚠️  Wan2.2 model exists but no inference script found")
    else:
        print("ℹ️  Run 'python setup_models.py' to download Wan2.2")

if __name__ == "__main__":
    main()