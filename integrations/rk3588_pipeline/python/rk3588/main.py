from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from common.config import load_app_config
from common.pipeline import StereoRuntimePipeline
from rk3588.rknn_backend import RknnLiteStereoBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightStereo RK3588 Python runtime")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "runtime" / "rk3588_python.json"),
        help="Path to the runtime config JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    backend = RknnLiteStereoBackend(config)
    pipeline = StereoRuntimePipeline(config, backend)
    pipeline.run()


if __name__ == "__main__":
    main()
