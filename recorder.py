import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import queue
import os

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data = []
        self._stream = None

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Error in audio stream: {status}")
        if self.recording:
            self.audio_data.append(indata.copy())

    def start(self):
        print("Starting recording...")
        self.audio_data = []
        self.recording = True
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self._callback)
        self._stream.start()

    def stop(self):
        print("Stopping recording...")
        self.recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        
        if not self.audio_data:
            return None
            
        return np.concatenate(self.audio_data, axis=0)

    def save_wav(self, data, filename="temp_recording.wav"):
        if data is None:
            return None
        # Ensure data is float32 for consistency, though sounddevice usually provides it
        # wav.write expects int16 or float32.
        # If it's float32, it should be in range [-1, 1]
        wav.write(filename, self.sample_rate, data.astype(np.float32))
        return os.path.abspath(filename)
