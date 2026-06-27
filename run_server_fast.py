import atexit
import faulthandler
import os
import signal
import sys
import time
import traceback

_log_dir = '/home/l/debug_panel/runtime_outputs'
os.makedirs(_log_dir, exist_ok=True)
_fatal_log_path = os.path.join(_log_dir, 'server_fatal.log')
_fatal_log = open(_fatal_log_path, 'a', buffering=1)
faulthandler.enable(file=_fatal_log, all_threads=True)


def _write_fatal(message):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    _fatal_log.write(f'[{ts}] {message}\n')
    _fatal_log.flush()


def _excepthook(exc_type, exc, tb):
    _write_fatal('unhandled exception')
    traceback.print_exception(exc_type, exc, tb, file=_fatal_log)
    _fatal_log.flush()
    sys.__excepthook__(exc_type, exc, tb)


def _signal_handler(signum, frame):
    _write_fatal(f'signal received: {signum}')
    raise SystemExit(128 + signum)


sys.excepthook = _excepthook
for _sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_sig, _signal_handler)
atexit.register(lambda: _write_fatal('process exiting'))

from app.bootstrap import prepare_runtime

prepare_runtime()

from app.server import flask_app, state as _unused
import app.server as sm
from app.main import create_app

print('[startup] create_app starting...', flush=True)
state = create_app()
print('[startup] create_app OK', flush=True)

sm.state = state
print('[startup] state assigned, starting Flask...', flush=True)

flask_app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
