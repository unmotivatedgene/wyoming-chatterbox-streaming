"""HTTP server with web interface for Chatterbox TTS."""

import argparse
import asyncio
import io
import json
import logging
import uuid
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from aiohttp import web
from aiohttp.web_exceptions import HTTPBadRequest

from .chatterbox import ChatterboxTTS
from . import handler as handler_module
from .handler import SUPPORTED_EXTENSIONS

_LOGGER = logging.getLogger(__name__)


class ChatterboxHTTPServer:
    """HTTP server with web interface for Chatterbox TTS."""

    def __init__(
        self,
        cli_args: argparse.Namespace,
        prompts_dir: Path,
    ):
        self.cli_args = cli_args
        self.prompts_dir = prompts_dir
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        """Set up HTTP routes."""
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/api/info", self.handle_info)
        self.app.router.add_post("/api/tts", self.handle_tts)
        self.app.router.add_post("/api/voices/upload", self.handle_voice_upload)
        self.app.router.add_delete("/api/voices/{name}", self.handle_voice_delete)
        self.app.router.add_get("/api/voices", self.handle_list_voices)
        self.app.router.add_post("/api/voices/{name}/test", self.handle_test_voice)

    async def handle_index(self, request: web.Request) -> web.Response:
        """Serve the main web interface."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wyoming Chatterbox</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; color: #eee; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #00d9ff; margin-bottom: 20px; }
        h2 { color: #00d9ff; margin: 30px 0 15px; font-size: 1.2rem; }
        .card {
            background: #16213e; border-radius: 12px; padding: 20px; margin-bottom: 20px;
        }
        label { display: block; margin-bottom: 8px; color: #aaa; }
        input, select, textarea {
            width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #333;
            background: #0f0f23; color: #eee; font-size: 16px; margin-bottom: 15px;
        }
        textarea { min-height: 100px; resize: vertical; }
        button {
            background: #00d9ff; color: #1a1a2e; border: none; padding: 12px 24px;
            border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer;
            transition: background 0.2s;
        }
        button:hover { background: #00b8d4; }
        button:disabled { background: #555; cursor: not-allowed; }
        .btn-danger { background: #ff4757; }
        .btn-danger:hover { background: #ff6b81; }
        .btn-secondary { background: #57606f; }
        .btn-secondary:hover { background: #747d8c; }
        #audioPlayer { width: 100%; margin-top: 15px; }
        .voice-list { display: grid; gap: 10px; }
        .voice-item {
            display: flex; justify-content: space-between; align-items: center;
            background: #0f0f23; padding: 15px; border-radius: 8px;
        }
        .voice-name { font-weight: bold; color: #00d9ff; }
        .voice-actions { display: flex; gap: 10px; }
        .voice-actions button { padding: 8px 12px; font-size: 14px; }
        .status { padding: 15px; border-radius: 8px; margin-bottom: 15px; }
        .status.loading { background: #ffd93d; color: #1a1a2e; }
        .status.error { background: #ff4757; color: white; }
        .status.success { background: #2ed573; color: #1a1a2e; }
        .file-input-wrapper { position: relative; margin-bottom: 15px; }
        .file-input-wrapper input[type="file"] { margin-bottom: 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Wyoming Chatterbox</h1>
        
        <div class="card">
            <h2>Model Status</h2>
            <div id="modelStatus">
                <span class="status loading">Loading model...</span>
            </div>
        </div>

        <div class="card">
            <h2>Synthesize Speech</h2>
            <label for="text">Text to synthesize</label>
            <textarea id="text" placeholder="Enter text to synthesize...">Hello! This is a test of the Chatterbox TTS system.</textarea>
            
            <label for="voice">Voice</label>
            <select id="voice">
                <option value="default">Default Voice</option>
            </select>
            
            <button id="synthesizeBtn" onclick="synthesize()">Generate Speech</button>
            <audio id="audioPlayer" controls style="display:none;"></audio>
        </div>

        <div class="card">
            <h2>Voice Cloning</h2>
            <p style="color: #888; margin-bottom: 15px;">
                Upload an audio clip (WAV, MP3, FLAC, OGG, M4A) to clone a voice. 
                Recommended: 10-30 seconds of clear speech.
            </p>
            <div class="file-input-wrapper">
                <input type="file" id="voiceFile" accept=".wav,.mp3,.flac,.ogg,.m4a">
            </div>
            <label for="voiceName">Voice Name</label>
            <input type="text" id="voiceName" placeholder="my-voice">
            <button onclick="uploadVoice()">Upload Voice</button>
        </div>

        <div class="card">
            <h2>Available Voices</h2>
            <div id="voiceList" class="voice-list">
                <p style="color: #888;">Loading voices...</p>
            </div>
        </div>
    </div>

    <script>
        let currentVoices = [];

        async function loadVoices() {
            try {
                const resp = await fetch('/api/voices');
                const voices = await resp.json();
                currentVoices = voices;
                
                const select = document.getElementById('voice');
                select.innerHTML = '<option value="default">Default Voice</option>';
                voices.forEach(v => {
                    const opt = document.createElement('option');
                    opt.value = v.name;
                    opt.textContent = v.name;
                    select.appendChild(opt);
                });

                const list = document.getElementById('voiceList');
                if (voices.length === 0) {
                    list.innerHTML = '<p style="color: #888;">No custom voices yet. Upload one above!</p>';
                } else {
                    list.innerHTML = voices.map(v => `
                        <div class="voice-item">
                            <span class="voice-name">${v.name}</span>
                            <div class="voice-actions">
                                <button class="btn-secondary" onclick="testVoice('${v.name}')">Test</button>
                                <button class="btn-danger" onclick="deleteVoice('${v.name}')">Delete</button>
                            </div>
                        </div>
                    `).join('');
                }
            } catch (e) {
                console.error('Failed to load voices:', e);
            }
        }

        async function loadModelStatus() {
            try {
                const resp = await fetch('/api/info');
                const info = await resp.json();
                const status = document.getElementById('modelStatus');
                status.innerHTML = '<span class="status success">✓ Model loaded and ready</span>';
            } catch (e) {
                const status = document.getElementById('modelStatus');
                status.innerHTML = '<span class="status error">✗ Model not loaded</span>';
            }
        }

        async function synthesize() {
            const text = document.getElementById('text').value;
            const voice = document.getElementById('voice').value;
            const btn = document.getElementById('synthesizeBtn');
            const audioPlayer = document.getElementById('audioPlayer');

            if (!text.trim()) {
                alert('Please enter some text');
                return;
            }

            btn.disabled = true;
            btn.textContent = 'Generating...';

            try {
                const resp = await fetch('/api/tts', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text, voice})
                });

                if (!resp.ok) {
                    const err = await resp.text();
                    throw new Error(err);
                }

                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                audioPlayer.src = url;
                audioPlayer.style.display = 'block';
                audioPlayer.play();
            } catch (e) {
                alert('Error: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate Speech';
            }
        }

        async function uploadVoice() {
            const file = document.getElementById('voiceFile').files[0];
            const name = document.getElementById('voiceName').value.trim();

            if (!file) {
                alert('Please select an audio file');
                return;
            }
            if (!name) {
                alert('Please enter a voice name');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);
            formData.append('name', name);

            try {
                const resp = await fetch('/api/voices/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!resp.ok) {
                    const err = await resp.text();
                    throw new Error(err);
                }

                alert('Voice uploaded successfully!');
                document.getElementById('voiceFile').value = '';
                document.getElementById('voiceName').value = '';
                loadVoices();
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        async function testVoice(name) {
            const text = document.getElementById('text').value || 'This is a test of the voice.';
            const voice = document.getElementById('voice').value;
            
            document.getElementById('voice').value = name;
            await synthesize();
            document.getElementById('voice').value = voice;
        }

        async function deleteVoice(name) {
            if (!confirm(`Delete voice "${name}"?`)) return;

            try {
                const resp = await fetch(`/api/voices/${name}`, {method: 'DELETE'});
                if (!resp.ok) {
                    const err = await resp.text();
                    throw new Error(err);
                }
                loadVoices();
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        loadModelStatus();
        loadVoices();
    </script>
</body>
</html>"""
        return web.Response(text=html, content_type="text/html")

    async def handle_info(self, request: web.Request) -> web.Response:
        """Return server info."""
        voices = self._discover_voices()
        
        info = {
            "name": "wyoming-chatterbox",
            "version": "1.0.0",
            "model_loaded": handler_module._MODEL is not None,
            "device": self.cli_args.device,
            "voices": voices,
        }
        return web.json_response(info)

    async def handle_tts(self, request: web.Request) -> web.Response:
        """Synthesize speech via HTTP with streaming response."""
        try:
            data = await request.json()
        except Exception:
            raise HTTPBadRequest(text="Invalid JSON")

        text = data.get("text", "").strip()
        if not text:
            raise HTTPBadRequest(text="Text is required")

        voice_name = data.get("voice", "default")
        audio_prompt_path = self._get_audio_prompt_path(voice_name)

        # Load model if needed
        async with handler_module._MODEL_LOCK:
            if handler_module._MODEL is None:
                _LOGGER.info("Loading Chatterbox-Turbo model...")
                handler_module._MODEL = ChatterboxTTS.from_pretrained(
                    device=self.cli_args.device,
                    hf_token=getattr(self.cli_args, 'hf_token', None)
                )
                _LOGGER.info("Model loaded successfully")

            _LOGGER.debug(f"Generating audio for: {text[:50]}...")
            wav = handler_module._MODEL.generate(text, audio_prompt_path=audio_prompt_path)

        # Convert to numpy
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()

        if wav.ndim > 1:
            wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav.squeeze()

        sample_rate = handler_module._MODEL.sr
        wav = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

        # Stream audio in chunks using chunked transfer encoding
        async def stream_audio():
            # Write WAV header first
            import struct
            
            # RIFF header
            data_size = len(wav.tobytes())
            wav_size = 36 + data_size
            
            yield b'RIFF'
            yield struct.pack('<I', wav_size)
            yield b'WAVE'
            
            # fmt chunk
            yield b'fmt '
            yield struct.pack('<I', 16)  # chunk size
            yield struct.pack('<H', 1)   # audio format (PCM)
            yield struct.pack('<H', 1)   # channels (mono)
            yield struct.pack('<I', sample_rate)  # sample rate
            yield struct.pack('<I', sample_rate * 2)  # byte rate
            yield struct.pack('<H', 2)   # block align
            yield struct.pack('<H', 16)  # bits per sample
            
            # data chunk
            yield b'data'
            yield struct.pack('<I', data_size)
            
            # Stream audio data in chunks
            chunk_size = 8192
            audio_bytes = wav.tobytes()
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i + chunk_size]

        return web.Response(
            body=stream_audio(),
            content_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )

    async def handle_voice_upload(self, request: web.Request) -> web.Response:
        """Upload a voice audio clip."""
        try:
            reader = await request.multipart()
        except Exception:
            raise HTTPBadRequest(text="Invalid multipart request")

        file = None
        name = None

        async for field in reader:
            if field.name == "file":
                file = field
            elif field.name == "name":
                name = await field.text()

        if not file or not name:
            raise HTTPBadRequest(text="File and name are required")

        # Sanitize name
        name = "".join(c for c in name if c.isalnum() or c in "-_").strip()
        if not name:
            raise HTTPBadRequest(text="Invalid name")

        # Determine file extension
        filename = file.filename or ""
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            ext = ".wav"

        # Save file
        voice_path = self.prompts_dir / f"{name}{ext}"
        
        # Read and save file content
        content = await file.read()
        voice_path.write_bytes(content)

        _LOGGER.info(f"Uploaded voice: {voice_path}")

        return web.json_response({"success": True, "name": name, "path": str(voice_path)})

    async def handle_voice_delete(self, request: web.Request) -> web.Response:
        """Delete a voice audio clip."""
        name = request.match_info["name"]

        deleted = False
        for ext in SUPPORTED_EXTENSIONS:
            voice_path = self.prompts_dir / f"{name}{ext}"
            if voice_path.exists():
                voice_path.unlink()
                deleted = True
                _LOGGER.info(f"Deleted voice: {voice_path}")
                break

        if not deleted:
            raise HTTPBadRequest(text=f"Voice '{name}' not found")

        return web.json_response({"success": True})

    async def handle_list_voices(self, request: web.Request) -> web.Response:
        """List available voices."""
        voices = self._discover_voices()
        return web.json_response(voices)

    async def handle_test_voice(self, request: web.Request) -> web.Response:
        """Test a specific voice."""
        name = request.match_info["name"]
        try:
            data = await request.json()
        except Exception:
            data = {}

        text = data.get("text", "This is a test of the voice cloning system.")
        audio_prompt_path = self._get_audio_prompt_path(name)

        async with handler_module._MODEL_LOCK:
            if handler_module._MODEL is None:
                handler_module._MODEL = ChatterboxTTS.from_pretrained(
                    device=self.cli_args.device,
                    hf_token=getattr(self.cli_args, 'hf_token', None)
                )

            wav = handler_module._MODEL.generate(text, audio_prompt_path=audio_prompt_path)

        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()
        if wav.ndim > 1:
            wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav.squeeze()

        sample_rate = handler_module._MODEL.sr
        wav = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(wav.tobytes())

        buffer.seek(0)
        return web.Response(
            body=buffer.read(),
            content_type="audio/wav",
        )

    def _discover_voices(self) -> list:
        """Discover available voices from prompts directory."""
        voices = []

        if self.prompts_dir.exists():
            for ext in SUPPORTED_EXTENSIONS:
                for wav_file in self.prompts_dir.glob(f"*{ext}"):
                    voice_name = wav_file.stem
                    voices.append({
                        "name": voice_name,
                        "file": wav_file.name,
                        "path": str(wav_file),
                    })

        return voices

    def _get_audio_prompt_path(self, voice_name: str) -> Optional[str]:
        """Get the audio prompt path for a given voice."""
        if voice_name != "default":
            for ext in SUPPORTED_EXTENSIONS:
                prompt_path = self.prompts_dir / f"{voice_name}{ext}"
                if prompt_path.exists():
                    return str(prompt_path)

        if self.cli_args.audio_prompt:
            default_path = Path(self.cli_args.audio_prompt)
            if default_path.exists():
                return str(default_path)

        for ext in SUPPORTED_EXTENSIONS:
            default_prompt = self.prompts_dir / f"default{ext}"
            if default_prompt.exists():
                return str(default_prompt)

        return None

    def run(self, host: str = "0.0.0.0", port: int = 5000):
        """Run the HTTP server."""
        web.run_app(self.app, host=host, port=port, print=None)
