from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
AUDIO_CHUNK_BYTES = 640


def pcm_rms(payload: bytes) -> int:
    if len(payload) < SAMPLE_WIDTH_BYTES:
        return 0
    sample_count = len(payload) // SAMPLE_WIDTH_BYTES
    samples = struct.unpack("<" + "h" * sample_count, payload[: sample_count * SAMPLE_WIDTH_BYTES])
    return int(math.sqrt(sum(sample * sample for sample in samples) / sample_count))


def read_wav_as_pcm(path: str | Path) -> bytes:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != CHANNELS:
            raise RuntimeError(f"WAV must be mono, got {wav_file.getnchannels()} channels")
        if wav_file.getsampwidth() != SAMPLE_WIDTH_BYTES:
            raise RuntimeError(f"WAV must be 16-bit, got sample width {wav_file.getsampwidth()}")
        if wav_file.getframerate() != SAMPLE_RATE:
            raise RuntimeError(f"WAV must be {SAMPLE_RATE} Hz, got {wav_file.getframerate()}")
        return wav_file.readframes(wav_file.getnframes())


def write_pcm_to_wav(path: str | Path, pcm: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm)
