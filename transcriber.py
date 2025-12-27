from faster_whisper import WhisperModel
import os
import torch

class Transcriber:
    def __init__(self, model_size="turbo", device=None):
        # Check if CUDA is available and requested
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        elif device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # faster-whisper uses different naming for some models or specific quantization
        model_name = model_size
        if model_size == "turbo":
            model_name = "deepdml/faster-whisper-large-v3-turbo-ct2"
        
        print(f"Loading Faster-Whisper model '{model_name}' on {device}...")
        
        # compute_type optimization:
        # CUDA: float16 is much faster and uses less VRAM.
        # CPU: int8 is faster than float32.
        compute_type = "float16" if device == "cuda" else "int8"
        
        try:
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=os.path.join(os.getcwd(), "models")
            )
        except Exception as e:
            print(f"Warning: Could not initialize model on {device}: {e}")
            if device == "cuda":
                print("Falling back to CPU with int8 quantization...")
                self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
            else:
                raise e

    def transcribe(self, audio_path, language=None):
        if not os.path.exists(audio_path):
            return ""
        
        print(f"Transcribing {audio_path}...")
        # beam_size=1 is much faster for local CPU usage
        segments, info = self.model.transcribe(audio_path, beam_size=1, language=language)
        
        text = "".join([segment.text for segment in segments])
        return text.strip()
