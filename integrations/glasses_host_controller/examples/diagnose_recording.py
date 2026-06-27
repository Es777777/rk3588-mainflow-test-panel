import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_utils import pcm_rms, write_pcm_to_wav
from glasses_device import DeviceEvent, GlassesDevice


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose ESP32-C3 microphone upload")
    parser.add_argument("--port")
    parser.add_argument("--out-dir", default="recordings")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--voice-rms", type=int, default=1800)
    parser.add_argument("--silence-seconds", type=float, default=2.0)
    parser.add_argument("--no-auto-stop", action="store_true")
    parser.add_argument("--host-start", action="store_true", help="Start recording by serial command instead of TALK")
    args = parser.parse_args()

    dev = GlassesDevice.open(args.port)
    pcm = bytearray()
    recording = False
    record_started_at = 0.0
    last_voice_at = 0.0
    last_report_at = time.monotonic()
    last_report_bytes = 0
    audio_packets = 0
    stop_sent = False

    def save() -> None:
        if not pcm:
            print("DIAG save skipped: no PCM bytes received", flush=True)
            return
        path = Path(args.out_dir) / f"diagnose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        write_pcm_to_wav(path, bytes(pcm))
        seconds = len(pcm) / 32000.0
        print(f"DIAG saved wav={path} bytes={len(pcm)} seconds={seconds:.2f}", flush=True)

    def on_audio(chunk: bytes) -> None:
        nonlocal audio_packets, last_voice_at, last_report_at, last_report_bytes
        if not recording:
            print(f"DIAG audio before record_start bytes={len(chunk)} rms={pcm_rms(chunk)}", flush=True)
            return

        audio_packets += 1
        pcm.extend(chunk)
        rms = pcm_rms(chunk)
        if rms >= args.voice_rms:
            last_voice_at = time.monotonic()

        now = time.monotonic()
        if now - last_report_at >= 1.0:
            bytes_delta = len(pcm) - last_report_bytes
            total_seconds = len(pcm) / 32000.0
            print(
                f"DIAG audio packets={audio_packets} total_bytes={len(pcm)} "
                f"total_audio_s={total_seconds:.2f} bytes_last_s={bytes_delta} rms={rms}",
                flush=True,
            )
            last_report_at = now
            last_report_bytes = len(pcm)

    def on_event(event: DeviceEvent) -> None:
        nonlocal recording, record_started_at, last_voice_at, stop_sent
        if event.kind == "hello":
            print(f"DIAG hello {event.data}", flush=True)
        elif event.kind == "button":
            print(f"DIAG button {event.data}", flush=True)
        elif event.kind == "log":
            print(f"DIAG log {event.data}", flush=True)
        elif event.kind == "status":
            print(f"DIAG status {event.data}", flush=True)
            if isinstance(event.data, dict) and event.data.get("recording") and not recording:
                pcm.clear()
                recording = True
                stop_sent = False
                record_started_at = time.monotonic()
                last_voice_at = record_started_at
                print("DIAG status says recording=true; counting incoming audio now", flush=True)
        elif event.kind == "record_start":
            pcm.clear()
            recording = True
            stop_sent = False
            record_started_at = time.monotonic()
            last_voice_at = record_started_at
            print("DIAG record_start received; speak now", flush=True)
        elif event.kind == "record_end":
            recording = False
            print(
                f"DIAG record_end received packets={audio_packets} bytes={len(pcm)}",
                flush=True,
            )
            save()
        elif event.kind == "disconnect":
            print(f"DIAG disconnect {event.data}", flush=True)

    dev.on_audio(on_audio)
    dev.on_event(on_event)
    dev.start()
    dev.get_status()

    if args.host_start:
        print("DIAG host-start requested; speak after the beep", flush=True)
        dev.start_recording()
        pcm.clear()
        recording = True
        stop_sent = False
        record_started_at = time.monotonic()
        last_voice_at = record_started_at
    else:
        print("DIAG ready. Press TALK on the glasses, speak, then wait for silence.", flush=True)
    deadline = time.monotonic() + args.seconds

    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if recording and not stop_sent and not args.no_auto_stop:
                if now - last_voice_at >= args.silence_seconds:
                    print("DIAG silence detected; sending stop_recording", flush=True)
                    stop_sent = True
                    dev.stop_recording()
            time.sleep(0.05)
    finally:
        if recording and not stop_sent:
            print("DIAG timeout while recording; sending stop_recording", flush=True)
            dev.stop_recording()
            time.sleep(1.5)
        dev.close()
        print("DIAG done", flush=True)


if __name__ == "__main__":
    main()
