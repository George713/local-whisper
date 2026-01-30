"""Audio utility functions for processing and conversion."""

import io
import logging
import numpy as np
import scipy.io.wavfile as wav

logger = logging.getLogger(__name__)


def save_to_bytes(audio, sample_rate):
    """Convert numpy array to WAV bytes for in-memory processing.
    
    Args:
        audio: numpy array of audio data (should be float32 in range [-1, 1])
        sample_rate: sample rate of the audio
        
    Returns:
        BytesIO buffer containing WAV data
    """
    if audio is None or len(audio) == 0:
        logger.warning("Empty audio provided to save_to_bytes")
        return io.BytesIO()
    
    # Ensure float32 format
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Create BytesIO buffer
    buffer = io.BytesIO()
    
    # Write WAV to buffer
    wav.write(buffer, sample_rate, audio)
    
    # Reset buffer position to beginning
    buffer.seek(0)
    
    return buffer


def get_duration(audio, sample_rate):
    """Calculate the duration of audio in seconds.
    
    Args:
        audio: numpy array of audio data or number of samples
        sample_rate: sample rate of the audio
        
    Returns:
        Duration in seconds as float
    """
    if audio is None:
        return 0.0
    
    # Handle both numpy arrays and integer sample counts
    if isinstance(audio, np.ndarray):
        num_samples = len(audio)
    else:
        num_samples = int(audio)
    
    if sample_rate <= 0:
        logger.warning(f"Invalid sample rate: {sample_rate}")
        return 0.0
    
    return num_samples / sample_rate


def load_from_bytes(buffer, dtype=np.float32):
    """Load audio from WAV bytes buffer.
    
    Args:
        buffer: BytesIO buffer or bytes containing WAV data
        dtype: desired output dtype (default: float32)
        
    Returns:
        Tuple of (audio data as numpy array, sample rate)
    """
    if isinstance(buffer, bytes):
        buffer = io.BytesIO(buffer)
    
    # Reset buffer position
    buffer.seek(0)
    
    # Read WAV data
    sample_rate, audio = wav.read(buffer)
    
    # Convert to float32 if needed
    if dtype == np.float32 and audio.dtype != np.float32:
        # wav.read returns int16, convert to float32 in range [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)
    
    return audio, sample_rate


def ensure_mono(audio):
    """Ensure audio is mono (single channel).
    
    Args:
        audio: numpy array of audio data (can be 1D or 2D)
        
    Returns:
        1D numpy array (mono audio)
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        # Average channels to get mono
        return audio.mean(axis=1)
    else:
        raise ValueError(f"Unsupported audio dimensions: {audio.ndim}")


def normalize_audio(audio, target_peak=0.9):
    """Normalize audio to target peak level.
    
    Args:
        audio: numpy array of audio data
        target_peak: target peak amplitude (default: 0.9)
        
    Returns:
        Normalized audio array
    """
    if audio is None or len(audio) == 0:
        return audio
    
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio * (target_peak / max_val)
    return audio
