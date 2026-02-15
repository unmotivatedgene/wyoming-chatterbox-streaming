# Wyoming Chatterbox Streaming

> **Created in conjunction with AI** 🤖

Wyoming protocol server for **Chatterbox-Turbo** TTS with **true streaming** support and web interface.

## License

MIT License - See LICENSE file for details.

**No Warranty**: This software is provided "as is", without warranty of any kind, express or implied.

## Acknowledgments & References

This project builds upon the excellent work of several open-source projects:

- **[wyoming-chatterbox](https://github.com/unmotivatedgene/wyoming-chatterbox)** - Original Wyoming protocol server for Chatterbox-Turbo
- **[wyoming-piper](https://github.com/rhasspy/wyoming-piper)** - Wyoming protocol implementation for Piper TTS (served as architecture reference)
- **[chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming)** - Streaming implementation inspiration

## Features

- **Wyoming Protocol**: Compatible with Home Assistant and Rhasspy
- **Chatterbox-Turbo**: Fast, low-latency TTS model (350M parameters)
- **True Streaming**: Audio chunks streamed as generated (~200ms to first chunk on GPU)
- **Voice Cloning**: Upload audio clips to create custom voices
- **Web Interface**: Test TTS, manage voices, and configure settings

## Quick Start

```bash
# Build
docker build -t wyoming-chatterbox-streaming .

# Run (CPU)
docker run -d -p 10200:10200 -p 5000:5000 \
  -v ./prompts:/data/prompts \
  wyoming-chatterbox-streaming --device cpu

# Run (GPU)
docker run -d --gpus all -p 10200:10200 -p 5000:5000 \
  -v ./prompts:/data/prompts \
  wyoming-chatterbox-streaming --device cuda
```

## Ports

- **10200/TCP**: Wyoming protocol (for Home Assistant)
- **5000/TCP**: Web interface and HTTP API

## Streaming Details

This implementation provides true streaming by:
1. Using Chatterbox-Turbo's optimized generation
2. Yielding audio chunks as they become available
3. Sending `AudioStart` immediately with first chunk

Expected performance on RTX 4090:
- Latency to first chunk: ~200ms
- Real-time factor: <1.0 (faster than real-time)

## Web Interface

Open http://localhost:5000 to:
- Test text-to-speech synthesis
- Upload audio clips for voice cloning
- Manage and delete custom voices

## Wyoming Protocol Usage

```python
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize

async with AsyncTcpClient("localhost", 10200) as client:
    await client.write_event(Synthesize(text="Hello!").event())
    # Audio chunks streamed as generated...
```

## Voice Cloning

1. Open the web interface at http://localhost:5000
2. Upload an audio file (WAV, MP3, FLAC, OGG, M4A)
3. Enter a name for the voice
4. Use the voice in synthesis

## Paralinguistic Tags

Chatterbox supports embedded audio tags for natural speech effects:

| Tag | Description | Example Usage |
|-----|-------------|---------------|
| `[clear throat]` | Throat clearing sound | Start of speech |
| `[sigh]` | Sighing | Expressing frustration |
| `[shush]` | Shushing sound | Quieting someone |
| `[cough]` | Coughing | Natural interruption |
| `[groan]` | Groaning | Expressing discomfort |
| `[sniff]` | Sniffing | Emotional or physical response |
| `[gasp]` | Gasping | Surprise or shock |
| `[chuckle]` | Light laughter | Mild amusement |
| `[laugh]` | Full laughter | Strong amusement |

Example: "Well, [sigh] I guess that's it. [laugh] Thanks for listening!"

## API

### POST /api/tts
```bash
curl -X POST http://localhost:5000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "my-voice"}' \
  --output speech.wav
```

### GET /api/info
Returns server status and available voices.

### GET /api/voices
List all available voice prompts.

## Configuration

### Environment Variables
- `HF_TOKEN`: HuggingFace token for model download (required first time)

### CLI Arguments
- `--device`: cuda or cpu (default: cuda)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Wyoming protocol port (default: 10200)
- `--http-port`: Web interface port (default: 5000)
- `--prompts-dir`: Directory for voice prompts (default: /data/prompts)
- `--audio-prompt`: Default audio prompt file
- `--samples-perChunk`: Samples per audio chunk (default: 10240)
- `--use-streaming`: Enable true streaming (default: true)
- `--chunk-size`: Speech tokens per chunk for streaming (default: 50)
- `--exaggeration`: Emotion intensity 0-1 (default: 0.5)
- `--cfg-weight`: CFG weight 0-1 (default: 0.5)

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
