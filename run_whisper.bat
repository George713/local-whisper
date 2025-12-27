@echo off
:: Local Whisper Launcher
:: This script starts the Local Whisper CLI using 'uv'.
:: Ensure you have 'uv' installed and have run 'uv sync' at least once.
:: Press the configured hotkey (default: Caps Lock) to start/stop recording.
uv run cli.py
pause
