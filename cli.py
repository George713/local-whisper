import yaml
import os
import time
import sys
import threading
from recorder import AudioRecorder
from transcriber import Transcriber
from keyboard_handler import KeyboardHandler

class LocalWhisperCLI:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.recorder = AudioRecorder()
        self.transcriber = Transcriber(
            model_size=self.config.get("model_size", "turbo"),
            device=self.config.get("device")
        )
        self.keyboard = KeyboardHandler(
            hotkey_name=self.config.get("hotkey", "caps_lock"),
            on_toggle_callback=self.toggle_recording
        )
        self.is_recording = False

    def _load_config(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def toggle_recording(self):
        # Run the toggle logic in a separate thread to avoid blocking the keyboard listener
        threading.Thread(target=self._toggle_recording_sync).start()

    def _toggle_recording_sync(self):
        if not self.is_recording:
            self.is_recording = True
            self.recorder.start()
            print(">>> RECORDING... (Press hotkey again to stop)")
        else:
            self.is_recording = False
            audio_data = self.recorder.stop()
            print(">>> PROCESSING...")
            
            if audio_data is not None:
                temp_file = self.recorder.save_wav(audio_data)
                try:
                    text = self.transcriber.transcribe(
                        temp_file,
                        language=self.config.get("language")
                    )
                    print(f">>> TRANSCRIPTION: {text}")
                    self.keyboard.inject_text(text, mode=self.config.get("paste_mode", "type"))
                except Exception as e:
                    print(f">>> Error during transcription: {e}")
                finally:
                    # Clean up
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            else:
                print(">>> No audio captured.")

    def run(self):
        print(f"Local Whisper CLI is running.")
        print(f"Hotkey: {self.config.get('hotkey')}")
        print("Press Ctrl+C to exit.")
        self.keyboard.start()
        try:
            while True:
                # On Windows, a short sleep in a loop is often more reliable
                # for catching KeyboardInterrupt than join()
                time.sleep(0.5)
                if not self.keyboard.listener.running:
                    break
        except (KeyboardInterrupt, SystemExit):
            print("\nExiting...")
        finally:
            self.keyboard.stop()
            # Force exit if threads are hanging
            os._exit(0)

if __name__ == "__main__":
    cli = LocalWhisperCLI()
    cli.run()
