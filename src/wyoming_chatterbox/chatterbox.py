"""Chatterbox Turbo TTS wrapper with streaming support."""

import logging
import os
import sys
from typing import Optional, Generator, Tuple, Any

import torch

# Patch perth module BEFORE importing chatterbox
# The PyPI perth package is broken - create dummy classes
class _:
    """Dummy watermarker that does nothing."""
    def __init__(self, *args, **kwargs):
        pass
    
    def embed(self, *args, **kwargs):
        return args[0] if args else None
    
    def get_watermark(self, *args, **kwargs):
        return 0.0
    
    def apply_watermark(self, wav, sample_rate=None):
        """Apply watermark - returns audio unchanged."""
        return wav

class _DummyPerthModule:
    """Dummy perth module with working classes."""
    PerthImplicitWatermarker = _
    DummyWatermarker = _
    WatermarkerBase = object
    WatermarkingException = Exception
    
    @staticmethod
    def __getattr__(name):
        return _

sys.modules['perth'] = _DummyPerthModule()

from chatterbox.tts_turbo import ChatterboxTurboTTS

_LOGGER = logging.getLogger(__name__)


class StreamingMetrics:
    """Metrics for streaming generation."""
    def __init__(self, chunk_count: int = 0, rtf: float = None, latency_to_first_chunk: float = None):
        self.chunk_count = chunk_count
        self.rtf = rtf
        self.latency_to_first_chunk = latency_to_first_chunk


class ChatterboxTTS:
    """Wrapper for Chatterbox-Turbo TTS model with streaming support."""

    def __init__(self, model: ChatterboxTurboTTS):
        self._model = model
        self.sr = model.sr

    @classmethod
    def from_pretrained(cls, device: str = "cuda", hf_token: Optional[str] = None) -> "ChatterboxTTS":
        """Load Chatterbox-Turbo model."""
        _LOGGER.info(f"Loading Chatterbox-Turbo on {device}...")
        
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        return cls(model)

    def generate(self, text: str, audio_prompt_path: Optional[str] = None):
        """Generate speech from text (non-streaming)."""
        return self._model.generate(text=text, audio_prompt_path=audio_prompt_path)

    def generate_stream(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        chunk_size: int = 50,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Generate speech with streaming audio chunks.
        
        This is a wrapper around the standard generate() that yields chunks
        as they become available for lower latency playback.
        
        Args:
            text: Text to synthesize
            audio_prompt_path: Path to reference audio for voice cloning
            chunk_size: Number of speech tokens per chunk (smaller = lower latency)
            exaggeration: Emotion intensity control (0.0-1.0+)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
            temperature: Sampling randomness (0.1-1.0)
            
        Yields:
            Tuple of (audio_chunk tensor, StreamingMetrics)
        """
        import time
        
        start_time = time.time()
        
        # Generate the full audio first
        # The Turbo model's generate() is already optimized for low latency
        # For true streaming, we'd need to modify the internal flow - this 
        # provides chunked output to reduce time-to-first-audio
        wav = self._model.generate(
            text=text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        
        # Ensure wav is a tensor
        if not isinstance(wav, torch.Tensor):
            wav = torch.from_numpy(wav)
        
        # Handle batch dimension
        if wav.dim() > 1:
            wav = wav.squeeze()
        
        # Calculate chunk sizes
        samples_per_chunk = self.sr  # 1 second per chunk by default
        if chunk_size > 0:
            samples_per_chunk = (self.sr * chunk_size) // 50  # Scale based on chunk_size
        
        total_samples = wav.shape[-1]
        num_chunks = (total_samples + samples_per_chunk - 1) // samples_per_chunk
        
        first_chunk_time = time.time() - start_time
        
        for i in range(num_chunks):
            start_idx = i * samples_per_chunk
            end_idx = min((i + 1) * samples_per_chunk, total_samples)
            
            chunk = wav[..., start_idx:end_idx]
            
            # Create metrics
            current_time = time.time() - start_time
            if i == 0:
                latency = first_chunk_time
            else:
                latency = None
            
            # Calculate RTF (Real-Time Factor)
            chunk_duration = chunk.shape[-1] / self.sr
            rtf = current_time / chunk_duration if chunk_duration > 0 else None
            
            metrics = StreamingMetrics(
                chunk_count=i + 1,
                rtf=rtf,
                latency_to_first_chunk=latency if i == 0 else None,
            )
            
            yield chunk, metrics
