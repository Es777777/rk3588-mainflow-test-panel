import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_utils import pcm_rms, write_pcm_to_wav
from glasses_device import DeviceEvent, GlassesDevice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--out-dir", default="recordings")
    parser.add_argument("--voice-rms", type=int, default=1200)
    parser.add_argument("--silence-seconds", type=float, default=2.0)
    args = parser.parse_args()

    dev = GlassesDevice.open(args.port, baud=args.baud)
    pcm = bytearray()
    recording = False
    last_voice_at = 0.0

    def save_recording() -> None:
        nonlocal pcm
        if not pcm:
            return
        path = Path(args.out_dir) / f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        write_pcm_to_wav(path, bytes(pcm))
        print(f"saved {path} bytes={len(pcm)}", flush=True)
        pcm.clear()

    def on_audio(chunk: bytes) -> None:
        nonlocal last_voice_at
        if not recording:
            return
        pcm.extend(chunk)
        if pcm_rms(chunk) >= args.voice_rms:
            import time

            last_voice_at = time.monotonic()

    def on_event(event: DeviceEvent) -> None:
        nonlocal recording, last_voice_at
        if event.kind == "record_start":
            import time

            recording = True
            last_voice_at = time.monotonic()
            pcm.clear()
            print("recording...", flush=True)
        elif event.kind == "record_end":
            recording = False
            save_recording()
        elif event.kind in {"hello", "log", "button"}:
            print(event.kind, event.data, flush=True)

    dev.on_audio(on_audio)
    dev.on_event(on_event)
    dev.start()

    try:
        import time

        while True:
            if recording and last_voice_at and time.monotonic() - last_voice_at >= args.silence_seconds:
                dev.stop_recording()
            time.sleep(0.05)
    except KeyboardInterrupt:
        dev.close()


if __name__ == "__main__":
    main()
