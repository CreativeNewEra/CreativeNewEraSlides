import pytest
import sys
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSettings


# Mock PyQt5 application for headless testing
@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication instance for testing PyQt5 components."""
    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()
    yield app
    # Note: Don't quit the app here as it might be used by other tests


@pytest.fixture
def mock_settings():
    """Mock QSettings to avoid file system interactions during testing."""
    with patch('PyQt5.QtCore.QSettings') as mock:
        settings_instance = MagicMock()
        mock.return_value = settings_instance
        yield settings_instance


@pytest.fixture
def mock_torch():
    """Mock torch operations to avoid GPU dependencies in tests."""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.cuda.empty_cache'), \
         patch('torch.cuda.memory_reserved', return_value=0), \
         patch('torch.cuda.memory_allocated', return_value=0):
        yield


@pytest.fixture
def mock_model_loading():
    """Mock AI model loading to avoid downloading models during tests."""
    mock_pipeline = MagicMock()
    mock_pipeline.to.return_value = mock_pipeline
    mock_pipeline.enable_model_cpu_offload.return_value = None
    
    with patch('diffusers.FluxPipeline.from_pretrained', return_value=mock_pipeline), \
         patch('diffusers.StableDiffusionPipeline.from_pretrained', return_value=mock_pipeline), \
         patch('diffusers.FluxPipeline.from_single_file', return_value=mock_pipeline), \
         patch('diffusers.StableDiffusionPipeline.from_single_file', return_value=mock_pipeline):
        yield mock_pipeline


@pytest.fixture
def mock_subprocess():
    """Mock subprocess operations for testing workers without external processes."""
    mock_process = MagicMock()
    mock_process.stdout = iter(["Progress: 50%", "Progress: 100%", "Complete"])
    mock_process.returncode = 0
    mock_process.poll.return_value = 0
    
    with patch('subprocess.Popen', return_value=mock_process):
        yield mock_process


@pytest.fixture
def mock_file_operations():
    """Mock file system operations for testing without creating actual files."""
    with patch('os.path.exists', return_value=True), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.is_dir', return_value=True), \
         patch('pathlib.Path.glob', return_value=[]):
        yield


@pytest.fixture
def mock_logging():
    """Mock logging to avoid log file creation during tests."""
    with patch('logging.getLogger') as mock_logger:
        logger_instance = MagicMock()
        mock_logger.return_value = logger_instance
        yield logger_instance


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Reset singleton instances between tests to ensure test isolation."""
    # Clear ModelManager singleton state
    from utils.model_manager import ModelManager
    ModelManager._flux_pipe = None
    ModelManager._flux_device = None
    ModelManager._settings_manager = None
    ModelManager._model_downloader = None
    
    yield
    
    # Clean up after test
    ModelManager._flux_pipe = None
    ModelManager._flux_device = None
    ModelManager._settings_manager = None
    ModelManager._model_downloader = None


@pytest.fixture
def sample_generation_params():
    """Provide sample parameters for testing AI generation."""
    return {
        'width': 512,
        'height': 512,
        'steps': 20,
        'guidance_scale': 7.5,
        'seed': 42,
        'model_path': '/fake/model/path',
        'device': 'cpu'
    }


@pytest.fixture
def sample_video_params():
    """Provide sample parameters for testing video generation."""
    from workers.params import VideoParams
    return VideoParams(
        width=480,
        height=480,
        frames=8,
        steps=6,
        precision='bfloat16',
        offload=True,
        t5_cpu=True
    )


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m "not slow"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m "not gpu"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark slow tests and GPU tests."""
    for item in items:
        # Mark tests that involve model loading as slow
        if "model" in item.name.lower() or "generation" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark tests that use CUDA as GPU tests
        if "cuda" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(pytest.mark.gpu)