# Local Whisper

Local Whisper is a CLI tool for real-time audio transcription and text injection. It allows you to record audio via a hotkey, transcribe it locally using OpenAI's Whisper models (via `faster-whisper`), and automatically inject the resulting text into your active application.

## Features

- **Local Transcription**: Uses `faster-whisper` for high-performance local inference.
- **Global Hotkey**: Toggle recording from anywhere in your system.
- **Text Injection**: Automatically types the transcribed text or copies it to the clipboard.
- **Configurable**: Easily adjust model size, device (CPU/CUDA), and language via a YAML file.

## Prerequisites

- **Python**: Version 3.10 or higher.
- **uv**: Recommended for dependency management.
- **FFmpeg**: Required by `faster-whisper` for audio processing. Ensure it is installed and added to your system's PATH.

## Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd local-whisper
    ```

2.  **Install dependencies**:
    Using `uv`:
    ```bash
    uv sync
    ```
    Or using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the application**:
    Edit the [`config.yaml`](config.yaml) file to suit your needs (see [Configuration](#configuration) below).

## Running the Application

To start the CLI tool, run:

```bash
uv run cli.py
```

Once running, use the configured hotkey (default: `caps_lock`) to start and stop recording.

## Configuration

The application is configured via [`config.yaml`](config.yaml). Below are the available options:

```yaml
# Configuration for local-whisper
hotkey: "caps_lock"  # Options: caps_lock, f9, f10, etc. (pynput Key names)
model_size: "base"   # Options: tiny, base, small, medium, large, turbo
device: "cpu"        # Options: cuda, cpu
language: null       # Set to a language code (e.g., "en", "de") or null for auto-detect
paste_mode: "type"   # Options: type (simulates typing), clipboard (ctrl+v)
```

- **hotkey**: The key used to toggle recording.
- **model_size**: The Whisper model variant to use. `turbo` or `base` are good starting points.
- **device**: Use `cuda` if you have a compatible NVIDIA GPU, otherwise use `cpu`.
- **language**: Force a specific language or leave as `null` for auto-detection.
- **paste_mode**: `type` will simulate keyboard input, while `clipboard` will copy the text (requires manual paste or system-specific handling).

## Compatibility

- **Windows 11**: This project has been primarily tested and verified on Windows 11.
- **Unix-based systems (Linux/macOS)**: While the core dependencies are cross-platform, it has not been extensively tested on Unix-based systems. Some adjustments to audio drivers or keyboard listeners might be necessary.

## License

[MIT License](LICENSE)
