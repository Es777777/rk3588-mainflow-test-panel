import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import time
import struct
import wave

import serial
import serial.tools.list_ports

from serial_protocol import (
    MSG_AUDIO_UP,
    MSG_AUDIO_DOWN,
    MSG_BUTTON_EVENT,
    MSG_HELLO,
    MSG_LOG,
    MSG_MODE_CHANGED,
    MSG_PLAY_DONE,
    MSG_PLAY_START,
    MSG_PLAY_STOP,
    MSG_RECORD_END,
    MSG_RECORD_START,
    FrameParser,
    build_frame,
    parse_button_event,
    parse_mode_changed,
)


SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
AUDIO_CHUNK_BYTES = 640
SILENCE_SECONDS = 2.0
VOICE_RMS_THRESHOLD = 1200
MAX_RECORD_SECONDS = 12.0


def choose_port(explicit_port: str | None) -> str:
    if explicit_port:
        return explicit_port

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found. Pass --port COMx after plugging in the ESP32-C3.")

    print("Available serial ports:")
    for port in ports:
        print(f"  {port.device}: {port.description}")

    return ports[0].device


def main() -> int:
    parser = argparse.ArgumentParser(description="ESP32-C3 glasses controller host test")
    parser.add_argument("--port", help="Serial port, for example COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, help="Optional run time in seconds")
    parser.add_argument("--play-wav", help="Send a 16kHz/16-bit/mono WAV to ESP32 for playback")
    parser.add_argument("--auto-repeat", action="store_true", help="Auto-stop recording after silence, save WAV, then play it back")
    args = parser.parse_args()

    try:
        port = choose_port(args.port)
        parser_state = FrameParser()
        recordings_dir = Path(__file__).resolve().parent / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        recording = bytearray()
        recording_started_at: datetime | None = None
        recording_started_mono: float | None = None
        tx_seq = 0
        playback_sent = False
        playback_done = False
        voice_seen = False
        last_voice_at = 0.0
        stop_recording_sent = False

        def pcm_rms(payload: bytes) -> int:
            if len(payload) < 2:
                return 0
            sample_count = len(payload) // 2
            samples = struct.unpack("<" + "h" * sample_count, payload[:sample_count * 2])
            return int(math.sqrt(sum(sample * sample for sample in samples) / sample_count))

        def send_host_frame(ser: serial.Serial, msg_type: int, payload: bytes = b"") -> None:
            nonlocal tx_seq
            ser.write(build_frame(msg_type, tx_seq, payload))
            ser.flush()
            tx_seq = (tx_seq + 1) & 0xFFFF

        def send_wav_for_playback(ser: serial.Serial, wav_path: Path) -> None:
            with wave.open(str(wav_path), "rb") as wav_file:
                if wav_file.getnchannels() != CHANNELS:
                    raise RuntimeError(f"WAV must be mono, got {wav_file.getnchannels()} channels")
                if wav_file.getsampwidth() != SAMPLE_WIDTH_BYTES:
                    raise RuntimeError(f"WAV must be 16-bit, got sample width {wav_file.getsampwidth()}")
                if wav_file.getframerate() != SAMPLE_RATE:
                    raise RuntimeError(f"WAV must be 16000 Hz, got {wav_file.getframerate()}")
                pcm = wav_file.readframes(wav_file.getnframes())

            print(f"PLAY_SEND path={wav_path} bytes={len(pcm)} seconds={len(pcm)/(SAMPLE_RATE*SAMPLE_WIDTH_BYTES):.2f}", flush=True)
            send_host_frame(ser, MSG_PLAY_START)
            for offset in range(0, len(pcm), AUDIO_CHUNK_BYTES):
                send_host_frame(ser, MSG_AUDIO_DOWN, pcm[offset:offset + AUDIO_CHUNK_BYTES])
                time.sleep(0.018)
            send_host_frame(ser, MSG_PLAY_STOP)
            print("PLAY_STOP sent", flush=True)

        print(f"Opening {port} at {args.baud} baud...", flush=True)
        with serial.Serial(port, args.baud, timeout=0.2) as ser:
            start = time.monotonic()
            end_at = time.monotonic() + args.duration if args.duration else None
            while True:
                data = ser.read(256)
                if data:
                    for frame in parser_state.feed(data):
                        if frame.msg_type == MSG_HELLO:
                            payload = frame.payload.decode("utf-8", errors="replace")
                            print(
                                f"ESP32 glasses controller connected: seq={frame.seq}, payload={payload}",
                                flush=True,
                            )
                            if args.play_wav and not playback_sent:
                                send_wav_for_playback(ser, Path(args.play_wav))
                                playback_sent = True
                            start = time.monotonic()
                            continue
                        if frame.msg_type == MSG_BUTTON_EVENT:
                            event = parse_button_event(frame.payload)
                            print(
                                f"BUTTON {event['button']} {event['event']} "
                                f"mode={event['mode']} ts={event['timestamp_ms']}ms",
                                flush=True,
                            )
                            start = time.monotonic()
                            continue
                        if frame.msg_type == MSG_MODE_CHANGED:
                            mode = parse_mode_changed(frame.payload)
                            print(
                                f"MODE_CHANGED mode={mode['mode']} "
                                f"source={mode['source']} ts={mode['timestamp_ms']}ms",
                                flush=True,
                            )
                            start = time.monotonic()
                            continue
                        if frame.msg_type == MSG_RECORD_START:
                            recording.clear()
                            recording_started_at = datetime.now()
                            recording_started_mono = time.monotonic()
                            voice_seen = False
                            last_voice_at = recording_started_mono
                            stop_recording_sent = False
                            print("RECORD_START", flush=True)
                            start = time.monotonic()
                            continue
                        if frame.msg_type == MSG_AUDIO_UP:
                            if recording_started_at is not None:
                                recording.extend(frame.payload)
                                rms = pcm_rms(frame.payload)
                                now = time.monotonic()
                                if rms >= VOICE_RMS_THRESHOLD:
                                    voice_seen = True
                                    last_voice_at = now
                                if len(recording) % 32000 < len(frame.payload):
                                    seconds = len(recording) / (SAMPLE_RATE * SAMPLE_WIDTH_BYTES)
                                    print(f"RECORDING bytes={len(recording)} seconds={seconds:.1f} rms={rms}", flush=True)
                                if (
                                    args.auto_repeat
                                    and not stop_recording_sent
                                    and (
                                        (voice_seen and now - last_voice_at >= SILENCE_SECONDS)
                                        or (
                                            recording_started_mono is not None
                                            and now - recording_started_mono >= MAX_RECORD_SECONDS
                                        )
                                    )
                                ):
                                    print("AUTO_SILENCE stopping recording", flush=True)
                                    send_host_frame(ser, MSG_RECORD_END)
                                    stop_recording_sent = True
                            start = time.monotonic()
                            continue
                        if frame.msg_type == MSG_RECORD_END:
                            wav_path = None
                            if recording_started_at is None:
                                print("RECORD_END without active recording", flush=True)
                            else:
                                timestamp = recording_started_at.strftime("%Y%m%d_%H%M%S")
                                wav_path = recordings_dir / f"record_{timestamp}.wav"
                                with wave.open(str(wav_path), "wb") as wav_file:
                                    wav_file.setnchannels(CHANNELS)
                                    wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
                                    wav_file.setframerate(SAMPLE_RATE)
                                    wav_file.writeframes(recording)
                                seconds = len(recording) / (SAMPLE_RATE * SAMPLE_WIDTH_BYTES)
                                print(f"RECORD_END saved={wav_path} bytes={len(recording)} seconds={seconds:.2f}", flush=True)
                                recording_started_at = None
                                recording_started_mono = None
                                recording.clear()
                            if args.auto_repeat and wav_path is not None:
                                send_wav_for_playback(ser, wav_path)
                            start = time.monotonic()
                            continue
                        if frame.msg_type == MSG_LOG:
                            message = frame.payload.decode("utf-8", errors="replace")
                            print(f"LOG {message}", flush=True)
                            start = time.monotonic()
                            continue
                        if frame.msg_type == MSG_PLAY_DONE:
                            print("PLAY_DONE", flush=True)
                            playback_done = True
                            if args.play_wav and args.duration is None:
                                return 0
                            start = time.monotonic()
                            continue
                        print(
                            f"Frame type=0x{frame.msg_type:02X}, seq={frame.seq}, "
                            f"payload_len={len(frame.payload)}",
                            flush=True,
                        )

                if end_at is not None and time.monotonic() >= end_at:
                    if recording_started_at is not None and not stop_recording_sent:
                        send_host_frame(ser, MSG_RECORD_END)
                        stop_recording_sent = True
                        end_at = time.monotonic() + 3
                        continue
                    return 0
                if args.play_wav and playback_sent and playback_done and args.duration is None:
                    return 0
    except KeyboardInterrupt:
        print("Stopped.")
        return 0
    except Exception as exc:
        print(f"Host test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
