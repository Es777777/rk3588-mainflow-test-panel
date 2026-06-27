import argparse
import time

import serial


def main() -> int:
    parser = argparse.ArgumentParser(description="Raw serial debug reader")
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    end_at = time.monotonic() + args.seconds
    with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
        while time.monotonic() < end_at:
            data = ser.read(256)
            if not data:
                continue
            printable = "".join(chr(b) if 32 <= b < 127 or b in (10, 13) else "." for b in data)
            print(f"HEX {data.hex(' ')}")
            print(f"TXT {printable}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
