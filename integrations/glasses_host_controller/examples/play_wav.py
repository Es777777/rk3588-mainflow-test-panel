import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glasses_device import GlassesDevice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=921600)
    args = parser.parse_args()

    dev = GlassesDevice.open(args.port, baud=args.baud)
    dev.play_wav(args.wav)
    dev.close()


if __name__ == "__main__":
    main()
