"""Wyoming event handler for Chatterbox TTS with streaming support."""

import argparse
import asyncio
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .chatterbox import ChatterboxTTS

_LOGGER = logging.getLogger(__name__)

# Global model instance
_MODEL: Optional[ChatterboxTTS] = None
_VOICE_NAME: Optional[str] = None
_MODEL_LOCK = asyncio.Lock()

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


class ChatterboxEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol TTS requests."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        prompts_dir: Path,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.prompts_dir = prompts_dir
        self.is_streaming: Optional[bool] = None
        self.sbd = SentenceBoundaryDetector()
        self._synthesize: Optional[Synthesize] = None

    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming protocol events."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        try:
            if Synthesize.is_type(event.type):
                return await self._handle_synthesize_event(event)

            if self.cli_args.no_streaming:
                return True

            if SynthesizeStart.is_type(event.type):
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                _LOGGER.debug("Text stream started: voice=%s", stream_start.voice)
                return True

            if SynthesizeChunk.is_type(event.type):
                assert self._synthesize is not None
                stream_chunk = SynthesizeChunk.from_event(event)
                
                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    _LOGGER.debug("Synthesizing stream sentence: %s", sentence)
                    self._synthesize.text = sentence
                    await self._handle_synthesize(self._synthesize)

                return True

            if SynthesizeStop.is_type(event.type):
                assert self._synthesize is not None
                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    await self._handle_synthesize(self._synthesize)

                await self.write_event(SynthesizeStopped().event())
                _LOGGER.debug("Text stream stopped")
                return True

            return True

        except Exception as err:
            _LOGGER.exception("Error handling event")
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_synthesize_event(self, event: Event) -> bool:
        """Handle a synthesize event (non-streaming)."""
        if self.is_streaming:
            return True

        synthesize = Synthesize.from_event(event)
        self.sbd = SentenceBoundaryDetector()
        
        start_sent = False
        for i, sentence in enumerate(self.sbd.add_chunk(synthesize.text)):
            self._synthesize = Synthesize(text=sentence, voice=synthesize.voice)
            await self._handle_synthesize(
                self._synthesize, send_start=(i == 0), send_stop=False
            )
            start_sent = True

        self._synthesize = Synthesize(text=self.sbd.finish(), voice=synthesize.voice)
        if self._synthesize.text:
            await self._handle_synthesize(
                self._synthesize, send_start=(not start_sent), send_stop=True
            )
        else:
            await self.write_event(AudioStop().event())

        return True

    async def _handle_synthesize(
        self,
        synthesize: Synthesize,
        send_start: bool = True,
        send_stop: bool = True,
    ) -> bool:
        """Synthesize text to audio and send via Wyoming protocol."""
        global _MODEL, _VOICE_NAME

        raw_text = synthesize.text
        text = " ".join(raw_text.strip().splitlines())

        if not text:
            return True

        # Resolve voice
        voice_name = "default"
        if synthesize.voice is not None and synthesize.voice.name:
            voice_name = synthesize.voice.name

        audio_prompt_path = self._get_audio_prompt_path(voice_name)

        # Load model if needed
        async with _MODEL_LOCK:
            if _MODEL is None:
                _LOGGER.info("Loading Chatterbox-Turbo model...")
                _MODEL = ChatterboxTTS.from_pretrained(
                    device=self.cli_args.device,
                    hf_token=getattr(self.cli_args, 'hf_token', None)
                )
                _LOGGER.info("Model loaded successfully")

            if voice_name != _VOICE_NAME:
                _LOGGER.debug(f"Voice changed to: {voice_name}")
                _VOICE_NAME = voice_name

            assert _MODEL is not None

            # Generate audio using streaming for lower latency
            _LOGGER.debug(f"Generating streaming audio for: {text[:50]}...")
            
            # Use streaming generation for better latency
            use_streaming = getattr(self.cli_args, 'use_streaming', True)
            
            if use_streaming and hasattr(_MODEL, 'generate_stream'):
                # Use streaming generation
                stream_start_time = time.time()
                first_chunk_sent = False
                
                for audio_chunk, metrics in _MODEL.generate_stream(
                    text,
                    audio_prompt_path=audio_prompt_path,
                    chunk_size=getattr(self.cli_args, 'chunk_size', 50),
                    exaggeration=getattr(self.cli_args, 'exaggeration', 0.5),
                    cfg_weight=getattr(self.cli_args, 'cfg_weight', 0.5),
                ):
                    # Convert to numpy
                    if isinstance(audio_chunk, torch.Tensor):
                        chunk_np = audio_chunk.squeeze().cpu().numpy()
                    else:
                        chunk_np = audio_chunk
                    
                    if chunk_np.ndim > 1:
                        chunk_np = chunk_np.mean(axis=0) if chunk_np.shape[0] > 1 else chunk_np.squeeze()
                    
                    # Convert to 16-bit PCM
                    chunk_np = (np.clip(chunk_np, -1.0, 1.0) * 32767).astype(np.int16)
                    
                    sample_rate = _MODEL.sr
                    
                    # Send AudioStart with first chunk
                    if not first_chunk_sent:
                        await self.write_event(
                            AudioStart(
                                rate=sample_rate,
                                width=2,
                                channels=1,
                            ).event()
                        )
                        first_chunk_sent = True
                        _LOGGER.debug(f"First chunk latency: {time.time() - stream_start_time:.3f}s")
                    
                    # Send the chunk
                    await self.write_event(
                        AudioChunk(
                            audio=chunk_np.tobytes(),
                            rate=sample_rate,
                            width=2,
                            channels=1,
                        ).event()
                    )
                
                # Send AudioStop
                if first_chunk_sent:
                    await self.write_event(AudioStop().event())
                
                return True
            else:
                # Fallback to non-streaming generation
                wav = _MODEL.generate(text, audio_prompt_path=audio_prompt_path)

        # Convert to numpy
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()
        
        if wav.ndim > 1:
            wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav.squeeze()
        
        sample_rate = _MODEL.sr

        # Convert to 16-bit PCM
        wav = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

        # Send audio start
        if send_start:
            await self.write_event(
                AudioStart(
                    rate=sample_rate,
                    width=2,
                    channels=1,
                ).event()
            )

        # Stream audio in chunks
        bytes_per_sample = 2
        channels = 1
        samples_per_chunk = self.cli_args.samples_per_chunk
        bytes_per_chunk = bytes_per_sample * channels * samples_per_chunk

        audio_bytes = wav.tobytes()
        num_chunks = math.ceil(len(audio_bytes) / bytes_per_chunk)

        for i in range(num_chunks):
            offset = i * bytes_per_chunk
            chunk = audio_bytes[offset : offset + bytes_per_chunk]

            await self.write_event(
                AudioChunk(
                    audio=chunk,
                    rate=sample_rate,
                    width=bytes_per_sample,
                    channels=channels,
                ).event()
            )

        if send_stop:
            await self.write_event(AudioStop().event())

        return True

    def _get_audio_prompt_path(self, voice_name: str) -> Optional[str]:
        """Get the audio prompt path for a given voice."""
        if voice_name != "default":
            for ext in SUPPORTED_EXTENSIONS:
                prompt_path = self.prompts_dir / f"{voice_name}{ext}"
                if prompt_path.exists():
                    _LOGGER.debug(f"Found voice prompt: {prompt_path}")
                    return str(prompt_path)

        if self.cli_args.audio_prompt:
            default_path = Path(self.cli_args.audio_prompt)
            if default_path.exists():
                return str(default_path)

        for ext in SUPPORTED_EXTENSIONS:
            default_prompt = self.prompts_dir / f"default{ext}"
            if default_prompt.exists():
                _LOGGER.debug(f"Found default prompt: {default_prompt}")
                return str(default_prompt)

        return None
