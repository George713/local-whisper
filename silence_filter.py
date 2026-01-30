"""Silero VAD-based silence filtering for audio processing."""

import io
import logging
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class SilenceFilter:
    """Silero VAD-based silence filtering implementation.
    
    Uses Silero VAD model with ONNX runtime for voice activity detection
    and silence trimming from audio recordings.
    """
    
    SILERO_SAMPLE_RATE = 16000  # Silero VAD requires 16kHz
    
    def __init__(self, config=None):
        """Initialize the SilenceFilter with configuration.
        
        Args:
            config: Dict with silence_filter configuration. If None, uses defaults.
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.threshold = self.config.get("threshold", 0.5)
        self.min_speech_duration_ms = self.config.get("min_speech_duration_ms", 250)
        self.min_silence_duration_ms = self.config.get("min_silence_duration_ms", 300)
        self.padding_ms = self.config.get("padding_ms", 100)
        
        self.model = None
        self._model_loaded = False
        
        if self.enabled:
            self._load_model()
    
    def _load_model(self):
        """Load the Silero VAD model with ONNX runtime."""
        try:
            logger.info("Loading Silero VAD model...")
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                onnx=True,
            )
            
            # Extract utility functions
            self.get_speech_timestamps = utils[0]
            self.save_audio = utils[1]
            self.read_audio = utils[2]
            self.VADIterator = utils[3]
            self.collect_chunks = utils[4]
            
            self._model_loaded = True
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            self._model_loaded = False
            self.enabled = False
    
    def _resample_if_needed(self, audio, sample_rate):
        """Resample audio to 16kHz if needed.
        
        Args:
            audio: numpy array of audio data
            sample_rate: current sample rate
            
        Returns:
            Tuple of (resampled audio, target sample rate)
        """
        if sample_rate != self.SILERO_SAMPLE_RATE:
            logger.debug(f"Resampling from {sample_rate}Hz to {self.SILERO_SAMPLE_RATE}Hz")
            
            # Convert numpy to torch tensor
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio.float()
            
            # Ensure mono (1 channel)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 2 and audio_tensor.shape[0] > 1:
                audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
            elif audio_tensor.dim() == 2 and audio_tensor.shape[1] > 1:
                audio_tensor = audio_tensor.mean(dim=1, keepdim=True)
            
            # Resample
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.SILERO_SAMPLE_RATE
            )
            audio_resampled = resampler(audio_tensor)
            
            # Convert back to numpy
            return audio_resampled.squeeze().numpy(), self.SILERO_SAMPLE_RATE
        
        return audio, sample_rate
    
    def _prepare_audio_for_vad(self, audio, sample_rate):
        """Prepare audio for VAD processing.
        
        Args:
            audio: numpy array of audio data
            sample_rate: current sample rate
            
        Returns:
            Tuple of (prepared torch tensor, effective sample rate)
        """
        # Resample if needed
        audio, effective_sr = self._resample_if_needed(audio, sample_rate)
        
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()
        
        # Ensure 1D tensor
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()
        
        return audio_tensor, effective_sr
    
    def has_speech(self, audio, sample_rate=16000):
        """Check if audio contains speech.
        
        Args:
            audio: numpy array of audio data
            sample_rate: sample rate of the audio
            
        Returns:
            Boolean indicating if speech was detected
        """
        if not self.enabled or not self._model_loaded:
            logger.debug("VAD disabled or model not loaded, assuming speech present")
            return True
        
        try:
            audio_tensor, effective_sr = self._prepare_audio_for_vad(audio, sample_rate)
            
            if len(audio_tensor) == 0:
                return False
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
            )
            
            return len(speech_timestamps) > 0
            
        except Exception as e:
            logger.error(f"Error in has_speech check: {e}")
            return True  # Assume speech on error to avoid losing data
    
    def filter_silence(self, audio, sample_rate=16000):
        """Filter silence from audio, keeping only speech segments.
        
        Args:
            audio: numpy array of audio data
            sample_rate: sample rate of the audio
            
        Returns:
            Filtered audio as numpy array (may be empty if no speech)
        """
        if not self.enabled or not self._model_loaded:
            logger.debug("VAD disabled or model not loaded, returning original audio")
            return audio
        
        try:
            audio_tensor, effective_sr = self._prepare_audio_for_vad(audio, sample_rate)
            
            if len(audio_tensor) == 0:
                logger.debug("Empty audio provided")
                return np.array([], dtype=np.float32)
            
            logger.debug("Detecting speech segments...")
            
            # Get speech timestamps with padding
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
            )
            
            if not speech_timestamps:
                logger.info("No speech detected in audio")
                return np.array([], dtype=np.float32)
            
            # Apply padding to timestamps
            padding_samples = int(self.padding_ms * effective_sr / 1000)
            for ts in speech_timestamps:
                ts["start"] = max(0, ts["start"] - padding_samples)
                ts["end"] = min(len(audio_tensor), ts["end"] + padding_samples)
            
            # Merge overlapping timestamps
            merged_timestamps = self._merge_timestamps(speech_timestamps)
            
            # Collect speech chunks
            speech_chunks = []
            for ts in merged_timestamps:
                start = ts["start"]
                end = ts["end"]
                speech_chunks.append(audio_tensor[start:end])
            
            if not speech_chunks:
                logger.info("No valid speech segments after processing")
                return np.array([], dtype=np.float32)
            
            # Concatenate all speech chunks
            filtered_audio = torch.cat(speech_chunks)
            
            logger.info(
                f"Filtered audio: {len(speech_timestamps)} speech segments, "
                f"original: {len(audio_tensor)/effective_sr:.2f}s, "
                f"filtered: {len(filtered_audio)/effective_sr:.2f}s"
            )
            
            return filtered_audio.numpy()
            
        except Exception as e:
            logger.error(f"Error filtering silence: {e}")
            logger.warning("Returning original audio due to filtering error")
            return audio
    
    def _merge_timestamps(self, timestamps):
        """Merge overlapping or adjacent timestamps.
        
        Args:
            timestamps: List of dicts with 'start' and 'end' keys
            
        Returns:
            List of merged timestamps
        """
        if not timestamps:
            return []
        
        # Sort by start time
        sorted_ts = sorted(timestamps, key=lambda x: x["start"])
        
        merged = [sorted_ts[0]]
        for current in sorted_ts[1:]:
            last = merged[-1]
            
            # Check if current overlaps or is adjacent to last
            if current["start"] <= last["end"]:
                # Merge by extending the end time
                last["end"] = max(last["end"], current["end"])
            else:
                merged.append(current)
        
        return merged
