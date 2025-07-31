# FluxWanApp

**A PyQt5 desktop app for fast, high-quality image (Flux) and video (Wan2.2) generation.**

## 🚀 Features

- **Image Generation** with Stable Diffusion (Flux) pipeline
- **Video Generation** via Wan2.2 CLI
- **Prompt History**: remembers your recent prompts (up to 10)
- **Drag & Drop**: load `.txt` prompts by dropping them into the window
- **Async Workers**: keeps the UI snappy during heavy generation
- **Auto-Save**: images saved with timestamped filenames in your output folder
- **Device Selection**: CPU or any available GPU (`cuda:0`, `cuda:1`, ...)

## 📁 Directory Structure

```
FluxWanApp/
├── ui/
│   └── main_window.py         # UI layout
├── workers/
│   └── image_and_video_workers.py  # Background threads
├── utils/
│   ├── settings_manager.py    # QSettings wrapper
│   ├── model_manager.py       # Model caching helpers
│   └── errors.py              # Friendly error parsing
├── controllers/
│   └── main_controller.py     # Wire UI ↔ workers
├── main.py                    # Entry point
├── requirements.txt           # Python deps
├── Dockerfile                 # Container instructions
├── .github/workflows/ci.yml   # CI pipeline
└── README.md
```

## 🛠 Prerequisites

- Linux (tested on Ubuntu 22.04 / Nobara)
- Python 3.10+
- NVIDIA GPU & drivers (optional, for GPU acceleration)
- Docker (optional, for containerized builds)

## 📦 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/FluxWanApp.git
   cd FluxWanApp
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Set model paths** (first run will prompt you to pick local model folders)

   ```bash
   # No interactive prompt yet—edit in Qt settings or use QSettings directly
   # e.g.,
   python3 - <<'EOF'
from utils.settings_manager import SettingsManager
s = SettingsManager()
s.set_model_path('flux', '/path/to/stable-diffusion')
EOF
   ```

4. **Run the app**
```bash
python3 main.py
```

## 🐳 Docker

Build and run with GPU support:

```bash
docker build -t fluxwanapp .
docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix fluxwanapp
```

## ⚙️ Usage

- **Image Tab**: Enter a prompt (or choose from history), tweak width/height/steps, and click **Generate Image**.
- **Video Tab**: Enter a prompt, set frames/steps, and click **Generate Video**.
- **Drag & Drop**: Drop a `.txt` file onto the window to load its contents into the image prompt.
- **History**: Select a past prompt from the dropdown or start typing to autocomplete.
- **Settings**: Model paths and output directory are stored via QSettings (persistent).

Generated images are saved to your configured output directory as:

```
flux_YYYYMMDD_HHMMSS.png
```

## 🧪 Testing & Linting

```bash
# Lint
black --check .
flake8 .
# Type-check
mypy .
# Run unit tests
pytest --maxfail=1 --disable-warnings -q
```

## 🤖 Continuous Integration

See `.github/workflows/ci.yml` for automated lint, type-check, tests, and build on every push/PR.

## 🎉 Next Steps

- Add a settings dialog for model paths & output folders
- Allow batch generation of multiple prompts
- Build an AppImage for easy Linux distribution
- Integrate video preview in-app instead of external player

Enjoy generating awesome imagery and videos—let creativity flow! 🚀

