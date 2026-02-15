#!/usr/bin/env python3
"""Wyoming Chatterbox - Main entry point."""

import argparse
import asyncio
import logging
import signal
from functools import partial
from pathlib import Path

import torch
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from . import __version__
from .handler import ChatterboxEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voice",
        default="default",
        help="Default voice to use (default: default)",
    )
    parser.add_argument(
        "--audio-prompt",
        help="Path to default audio prompt .wav file for voice cloning",
    )
    parser.add_argument(
        "--uri",
        default="tcp://0.0.0.0:10200",
        help="Wyoming URI (default: tcp://0.0.0.0:10200)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=5000,
        help="HTTP API port (0 to disable, default: 5000)",
    )
    parser.add_argument(
        "--data-dir",
        default="/data",
        help="Data directory for models and prompts (default: /data)",
    )
    parser.add_argument(
        "--prompts-dir",
        help="Directory containing voice prompt .wav files (default: data-dir/prompts)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference (default: cuda)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for downloading models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--samples-per-chunk",
        type=int,
        default=4096,
        help="Samples per audio chunk (default: 4096)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable audio streaming",
    )
    parser.add_argument(
        "--use-streaming",
        action="store_true",
        default=True,
        help="Enable true streaming (default: true)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Speech tokens per chunk for streaming (default: 50)",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Emotion intensity 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="CFG weight 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="chatterbox",
        help="Enable discovery over zeroconf with optional name",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log DEBUG messages",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _LOGGER.debug(args)

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        _LOGGER.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Set up directories
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else data_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Discover available voices
    voices = discover_voices(prompts_dir, args.voice)

    # Create Wyoming info
    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="chatterbox-turbo",
                description="Chatterbox-Turbo - State-of-the-art open-source TTS",
                attribution=Attribution(
                    name="Resemble AI",
                    url="https://github.com/resemble-ai/chatterbox",
                ),
                installed=True,
                voices=sorted(voices, key=lambda v: v.name),
                version=__version__,
                supports_synthesize_streaming=(not args.no_streaming),
            )
        ],
    )

    # Start Wyoming server
    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        from wyoming.zeroconf import HomeAssistantZeroconf

        if not hasattr(server, "port"):
            raise ValueError("Zeroconf requires TCP server")

        tcp_server = server
        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf,
            port=tcp_server.port,
            host=tcp_server.host,
        )
        await hass_zeroconf.register_server()
        _LOGGER.debug("Zeroconf discovery enabled")

    _LOGGER.info("Starting Wyoming Chatterbox server")
    _LOGGER.info(f"Device: {args.device}")
    _LOGGER.info(f"Default voice: {args.voice}")
    _LOGGER.info(f"Prompts directory: {prompts_dir}")

    # Start HTTP server if port is set
    http_server = None
    if args.http_port > 0:
        from .http_server import ChatterboxHTTPServer
        http_server = ChatterboxHTTPServer(args, prompts_dir)
        _LOGGER.info(f"HTTP server will start on port {args.http_port}")

    # Wyoming server task
    server_task = asyncio.create_task(
        server.run(
            partial(
                ChatterboxEventHandler,
                wyoming_info,
                args,
                prompts_dir,
            )
        )
    )

    # Handle signals
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, server_task.cancel)
    loop.add_signal_handler(signal.SIGTERM, server_task.cancel)

    # Start HTTP server in background
    if http_server:
        http_runner = None
        try:
            from aiohttp import web
            app = http_server.app
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", args.http_port)
            await site.start()
            _LOGGER.info(f"HTTP server started on port {args.http_port}")
            
            # Keep running
            await server_task
        except asyncio.CancelledError:
            _LOGGER.info("Server stopped")
    else:
        try:
            await server_task
        except asyncio.CancelledError:
            _LOGGER.info("Server stopped")


def discover_voices(prompts_dir: Path, default_voice: str) -> list:
    """Discover available voices from prompts directory."""
    voices = []
    supported_ext = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

    # Default voice always available
    voices.append(
        TtsVoice(
            name="default",
            description="Default Chatterbox-Turbo voice",
            attribution=Attribution(
                name="Resemble AI",
                url="https://github.com/resemble-ai/chatterbox",
            ),
            installed=True,
            languages=["en"],
            version=__version__,
        )
    )

    # Add custom voices from prompts directory
    if prompts_dir.exists():
        for ext in supported_ext:
            for wav_file in prompts_dir.glob(f"*{ext}"):
                voice_name = wav_file.stem
                if voice_name != default_voice:
                    voices.append(
                        TtsVoice(
                            name=voice_name,
                            description=f"Custom voice from {wav_file.name}",
                            attribution=Attribution(
                                name="User-provided",
                                url="",
                            ),
                            installed=True,
                            languages=["en"],
                            version=__version__,
                        )
                    )

    return voices


def run():
    """Run the main entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
