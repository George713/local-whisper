from faster_whisper import WhisperModel
import os
from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)

REQUIRED_MODEL_FILES = (
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
)


class Transcriber:
    def __init__(self, model_size="turbo", device=None, transcription_config=None):
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

        download_root = Path(os.getcwd()) / "models"
        cached_model_path = self._find_complete_cached_snapshot(model_name, download_root)
        if cached_model_path:
            print(
                f"Using cached Faster-Whisper model snapshot "
                f"'{cached_model_path.name}' for '{model_name}'."
            )
            model_name = str(cached_model_path)
        
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
                download_root=str(download_root)
            )
        except Exception as e:
            print(f"Warning: Could not initialize model on {device}: {e}")
            if device == "cuda":
                print("Falling back to CPU with int8 quantization...")
                self.model = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(download_root)
                )
            else:
                raise e
        
        # Store transcription configuration
        self.transcription_config = transcription_config or {}
        self.filter_no_speech = self.transcription_config.get("filter_no_speech", True)
        self.no_speech_threshold = self.transcription_config.get("no_speech_threshold", 0.6)

    def _find_complete_cached_snapshot(self, model_name, download_root):
        """Return a complete Hugging Face cache snapshot when one is already present."""
        if os.path.isdir(model_name) or "/" not in model_name:
            return None

        snapshots_dir = (
            download_root
            / f"models--{model_name.replace('/', '--')}"
            / "snapshots"
        )
        if not snapshots_dir.exists():
            return None

        complete_snapshots = [
            snapshot
            for snapshot in snapshots_dir.iterdir()
            if snapshot.is_dir()
            and all((snapshot / filename).exists() for filename in REQUIRED_MODEL_FILES)
        ]
        if not complete_snapshots:
            return None

        refs_main = snapshots_dir.parent / "refs" / "main"
        if refs_main.exists():
            try:
                current_revision = refs_main.read_text(encoding="utf-8").strip()
                for snapshot in complete_snapshots:
                    if snapshot.name == current_revision:
                        return snapshot
                logger.info(
                    "Current cached revision %s is incomplete; using complete cached snapshot instead",
                    current_revision,
                )
            except OSError as exc:
                logger.debug("Could not read cached model ref %s: %s", refs_main, exc)

        return max(
            complete_snapshots,
            key=lambda snapshot: (snapshot / "model.bin").stat().st_mtime,
        )

    def transcribe(self, audio_path, language=None):
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file does not exist: {audio_path}")
            return ""
        
        logger.info(f"Transcribing {audio_path}...")
        # beam_size=1 is much faster for local CPU usage
        segments, info = self.model.transcribe(audio_path, beam_size=1, language=language)
        
        # Convert generator to list to allow multiple iterations
        segments_list = list(segments)
        
        if not segments_list:
            logger.info("No segments found in audio")
            return ""
        
        # Filter segments based on no_speech_prob if enabled
        if self.filter_no_speech:
            filtered_segments = []
            removed_count = 0
            
            for segment in segments_list:
                if hasattr(segment, 'no_speech_prob') and segment.no_speech_prob >= self.no_speech_threshold:
                    removed_count += 1
                    logger.debug(
                        f"Filtering segment with high no_speech_prob: "
                        f"{segment.no_speech_prob:.2f} >= {self.no_speech_threshold}"
                    )
                    continue
                filtered_segments.append(segment)
            
            if removed_count > 0:
                logger.info(f"Filtered {removed_count} segments with no_speech_prob >= {self.no_speech_threshold}")
            
            segments_list = filtered_segments
        
        # Concatenate remaining segments for final text
        text = "".join([segment.text for segment in segments_list])
        
        logger.info(f"Transcription completed: {len(text)} characters from {len(segments_list)} segments")
        
        return text.strip()
