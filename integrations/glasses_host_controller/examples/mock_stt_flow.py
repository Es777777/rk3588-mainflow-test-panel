import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_utils import pcm_rms, write_pcm_to_wav
from glasses_device import DeviceEvent, GlassesDevice


def main() -> None:
    parser = argparse.ArgumentParser(description="Local mock STT/TTS flow for glasses bring-up")
    parser.add_argument("--port")
    parser.add_argument("--out-dir", default="recordings")
    parser.add_argument("--voice-rms", type=int, default=1200)
    parser.add_argument("--silence-seconds", type=float, default=2.0)
    parser.add_argument("--max-record-seconds", type=float, default=12.0)
    parser.add_argument("--mock-text", default="这是本地模拟的语音识别结果")
    parser.add_argument("--no-playback", action="store_true")
    args = parser.parse_args()

    dev = GlassesDevice.open(args.port)
    pcm = bytearray()
    recording = False
    record_started_at = 0.0
    last_voice_at = 0.0
    stop_sent = False

    def finish_recording() -> None:
        nonlocal recording, stop_sent
        if stop_sent:
            return
        stop_sent = True
        dev.stop_recording()

    def save_wav() -> Path | None:
        if not pcm:
            return None
        path = Path(args.out_dir) / f"mock_stt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        write_pcm_to_wav(path, bytes(pcm))
        return path

    def on_audio(chunk: bytes) -> None:
        nonlocal last_voice_at
        if not recording:
            return
        pcm.extend(chunk)
        if pcm_rms(chunk) >= args.voice_rms:
            last_voice_at = time.monotonic()

    def on_event(event: DeviceEvent) -> None:
        nonlocal recording, record_started_at, last_voice_at, stop_sent
        if event.kind == "hello":
            print(f"connected: {event.data}", flush=True)
        elif event.kind == "button":
            print(f"button: {event.data}", flush=True)
        elif event.kind == "record_start":
            pcm.clear()
            recording = True
            stop_sent = False
            record_started_at = time.monotonic()
            last_voice_at = record_started_at
            print("recording started", flush=True)
        elif event.kind == "record_end":
            recording = False
            wav_path = save_wav()
            print(f"recording ended, wav={wav_path}", flush=True)
            print(f"mock stt text: {args.mock_text}", flush=True)
            if wav_path is not None and not args.no_playback:
                print("playing recorded audio back as mock TTS output", flush=True)
                dev.play_wav(wav_path)
        elif event.kind == "log":
            print(f"log: {event.data}", flush=True)

    dev.on_audio(on_audio)
    dev.on_event(on_event)
    dev.start()

    try:
        while True:
            now = time.monotonic()
            if recording and not stop_sent:
                if now - last_voice_at >= args.silence_seconds:
                    print("silence detected, stopping recording", flush=True)
                    finish_recording()
                elif now - record_started_at >= args.max_record_seconds:
                    print("max recording length reached, stopping recording", flush=True)
                    finish_recording()
            time.sleep(0.05)
    except KeyboardInterrupt:
        dev.close()


if __name__ == "__main__":
    main()
