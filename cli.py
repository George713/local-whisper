import yaml
import os
import time
import sys
import threading
import logging
from recorder import AudioRecorder
from transcriber import Transcriber
from keyboard_handler import KeyboardHandler
from silence_filter import SilenceFilter
from audio_utils import get_duration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalWhisperCLI:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.recorder = AudioRecorder()
        self.transcriber = Transcriber(
            model_size=self.config.get("model_size", "turbo"),
            device=self.config.get("device"),
            transcription_config=self.config.get("transcription", {})
        )
        self.keyboard = KeyboardHandler(
            hotkey_name=self.config.get("hotkey", "caps_lock"),
            on_toggle_callback=self.toggle_recording
        )
        self.is_recording = False
        
        # Initialize silence filter with config
        silence_filter_config = self.config.get("silence_filter", {})
        self.silence_filter = SilenceFilter(config=silence_filter_config)
        
        # Get minimum recording duration
        self.min_recording_duration_sec = silence_filter_config.get(
            "min_recording_duration_sec", 0.5
        )

    def _load_config(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return {}
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
                # Check minimum recording duration
                duration = get_duration(audio_data, self.recorder.sample_rate)
                logger.info(f"Recording duration: {duration:.2f}s")
                
                if duration < self.min_recording_duration_sec:
                    print(f">>> Recording too short ({duration:.2f}s < {self.min_recording_duration_sec}s). Skipping.")
                    return
                
                # Apply silence filtering
                try:
                    filtered_audio = self.silence_filter.filter_silence(
                        audio_data,
                        sample_rate=self.recorder.sample_rate
                    )
                except Exception as e:
                    logger.error(f"Error during silence filtering: {e}")
                    filtered_audio = audio_data  # Fallback to original audio
                
                # Check if any speech remains after filtering
                if filtered_audio is None or len(filtered_audio) == 0:
                    print(">>> No speech detected in recording. Skipping transcription.")
                    return
                
                # Save filtered audio to temp file
                temp_file = self.recorder.save_wav(filtered_audio)
                
                try:
                    text = self.transcriber.transcribe(
                        temp_file,
                        language=self.config.get("language")
                    )
                    
                    if text and text.strip():
                        print(f">>> TRANSCRIPTION: {text}")
                        self.keyboard.inject_text(text, mode=self.config.get("paste_mode", "type"))
                    else:
                        print(">>> No transcription generated.")
                        
                except Exception as e:
                    logger.error(f"Error during transcription: {e}")
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
