import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glasses_device import GlassesDevice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port")
    args = parser.parse_args()

    dev = GlassesDevice.open(args.port)

    dev.on_button("MODE", "short", lambda event: print("switch app mode", event, flush=True))
    dev.on_button("TALK", "down", lambda event: print("recording starts on ESP32", event, flush=True))
    dev.on_button("TALK", "up", lambda event: print("recording stops on ESP32", event, flush=True))

    dev.run_forever()


if __name__ == "__main__":
    main()
