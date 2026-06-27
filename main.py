#!/usr/bin/env python3
import logging as _lg
_lg.disable(_lg.CRITICAL)

from app.bootstrap import prepare_runtime
from app.gui import run_native

if __name__ == '__main__':
    prepare_runtime()
    run_native()
