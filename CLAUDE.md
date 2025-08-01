# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CreativeNewEraSlides** is a PyQt5 desktop application for AI-powered image and video generation using Flux models (by Black Forest Labs) for image generation and Wan2.2 models (by Alibaba) for video generation. The application follows an MVC architecture with threaded workers for background AI processing.

## Development Commands

### Setup
```bash
# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install .[dev]

# Download required AI models (requires Hugging Face token)
python setup_models.py
```

### Code Quality and Testing
```bash
# Run all tests
pytest --maxfail=1 --disable-warnings -q

# Run specific test file
pytest tests/test_main_controller.py -v

# Code formatting
black .
black --check .  # Check only, don't modify

# Linting
flake8 .

# Type checking
mypy --ignore-missing-imports .

# Run all quality checks (same as CI)
black --check . && flake8 . && mypy --ignore-missing-imports . && pytest --maxfail=1 --disable-warnings -q
```

### Running the Application
```bash
# Quick start (handles venv activation)
./run.sh

# Or manually
source venv/bin/activate
python main.py
```

### Model Management
```bash
# Download/update models with your Hugging Face token
python setup_models.py

# Download specific model only
python -m utils.model_downloader --token YOUR_TOKEN --model flux
```

### Docker
```bash
# Build Docker image
docker build -t fluxwanapp .
```

## Architecture Overview

### Core Structure
- **MVC Pattern**: Controllers orchestrate UI and workers, UI handles PyQt5 interface, workers manage background AI processing
- **Threaded Architecture**: Heavy AI computations run in QThread instances to keep UI responsive
- **Signal-Slot Communication**: PyQt5 signals enable asynchronous communication between components

### Key Directories
- `controllers/` - Business logic layer (main_controller.py orchestrates the application)
- `ui/` - PyQt5 user interface definitions with tabbed layout
- `workers/` - Background thread implementations for AI model operations
- `utils/` - Shared utilities (settings, model management, error handling, logging)

### Important Patterns
- **Singleton Model Manager** (`utils/model_manager.py`) - Caches loaded models to avoid repeated loading
- **Settings Persistence** (`utils/settings_manager.py`) - Uses QSettings for configuration storage
- **Error Abstraction** (`utils/errors.py`) - Converts technical exceptions to user-friendly messages
- **Parameter Data Classes** (`workers/params.py`) - Structured data for AI generation parameters

## Technology Stack

### Core Dependencies
- **PyQt5 5.15.13** - Desktop GUI framework
- **PyTorch 2.0.1** - Deep learning model execution
- **Diffusers 0.34.0** - AI model pipeline management (supports Flux and Wan2.2 models)
- **Transformers 4.34.0** - Transformer model support
- **Accelerate 1.9.0** - Optimized model loading and inference

### Development Tools
- **pytest 8.4.1** - Testing framework with PyQt5 mocking
- **Black 25.1.0** - Code formatter (88 char line length, Python 3.8 target)
- **Flake8 7.3.0** - Linter (extends ignore for E203, W503)
- **MyPy 1.6.1** - Type checker (ignores missing imports)

## Testing Notes

- Tests use custom PyQt5 mocking to avoid GUI dependencies
- All tests are designed to run headlessly in CI environments
- Use `pytest --maxfail=1 --disable-warnings -q` for the same test configuration as CI
- Mock-based testing pattern for controllers and UI components

## AI Model Optimization for 16GB VRAM

### Flux Model Optimization
- **Model Variants**: Use FLUX.1-schnell for faster generation or FLUX.1-dev with quantization
- **Memory Techniques**: 
  - Enable group/leaf-level offloading with `apply_group_offloading()`
  - Use sequential CPU offload with `enable_sequential_cpu_offload()`
  - Enable VAE slicing/tiling: `vae.enable_slicing()` and `vae.enable_tiling()`
- **Precision**: Use bfloat16/FP16 with `torch_dtype=torch.bfloat16`
- **Quantization**: 4-bit/8-bit quantization with bitsandbytes for T5 encoder and transformer
- **Recommended Settings**: 576×1024 or 768×768 resolution, 20-30 steps, guidance scale 3-5

### Wan2.2 Video Model Optimization  
- **Model Selection**: Use TI2V-5B variant (5B parameters) for single-GPU setups
- **CLI Flags**: 
  - `--offload_model True` - offload model parts to CPU
  - `--convert_model_dtype bfloat16` - lower precision weights
  - `--t5_cpu` - run T5 text encoder on CPU
  - Disable `--enable_parallel_decode` for 16GB VRAM
- **Recommended Settings**: 480×480 or 640×360 resolution, 6-8 steps, 10-12 frames

### Memory Management
- Clear CUDA caches with `torch.cuda.empty_cache()` between generations
- Delete pipelines and call `gc.collect()` to free memory
- Monitor VRAM usage and warn users about memory limits

## Development Workflow

1. **Code Style**: Follow Black formatting (88 char line length) and Flake8 linting rules
2. **Type Hints**: Add type hints where beneficial, MyPy ignores missing imports
3. **Testing**: Write unit tests for new functionality, especially for core logic in controllers and utils
4. **Threading**: Use QThread for any CPU/GPU intensive operations to maintain UI responsiveness
5. **Settings**: Use the settings manager for any persistent configuration needs
6. **AI Models**: Follow the optimization guidelines above when working with Flux/Wan2.2 models