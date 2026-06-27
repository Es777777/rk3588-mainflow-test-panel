import atexit
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


_SERVER_PROCESS = None
_STARTED_SERVER = False
_SERVER_LOG_HANDLE = None
_SERVER_LOCK = threading.RLock()


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _server_url():
    return 'http://127.0.0.1:5000'


def _server_alive():
    try:
        urllib.request.urlopen(f'{_server_url()}/api/status', timeout=2)
        return True
    except Exception:
        return False


def _cleanup_server():
    global _SERVER_PROCESS, _SERVER_LOG_HANDLE
    if _SERVER_PROCESS is not None:
        if _SERVER_PROCESS.poll() is None:
            try:
                _SERVER_PROCESS.terminate()
                _SERVER_PROCESS.wait(timeout=3)
            except Exception:
                try:
                    _SERVER_PROCESS.kill()
                except Exception:
                    pass
    _SERVER_PROCESS = None
    if _SERVER_LOG_HANDLE is not None:
        try:
            _SERVER_LOG_HANDLE.close()
        except Exception:
            pass
        _SERVER_LOG_HANDLE = None


def _start_server_process():
    global _SERVER_PROCESS, _SERVER_LOG_HANDLE, _STARTED_SERVER
    repo = _repo_root()
    command = [sys.executable, str(repo / 'run_server_fast.py')]
    log_path = repo / 'runtime_outputs' / 'gui_server.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _SERVER_LOG_HANDLE = open(log_path, 'ab')
    _SERVER_PROCESS = subprocess.Popen(
        command,
        cwd=str(repo),
        stdout=_SERVER_LOG_HANDLE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    _STARTED_SERVER = True
    return _SERVER_PROCESS


def start_flask():
    global _SERVER_PROCESS, _STARTED_SERVER
    if _server_alive():
        return True
    with _SERVER_LOCK:
        _start_server_process()
    atexit.register(_cleanup_server)

    for _ in range(45):
        if _SERVER_PROCESS.poll() is not None:
            return False
        if _server_alive():
            return True
        time.sleep(1)
    return False


def _restart_server():
    with _SERVER_LOCK:
        _cleanup_server()
        _start_server_process()
    for _ in range(45):
        if _server_alive():
            return True
        time.sleep(1)
    return False


def _monitor_server(reload_callback=None):
    while True:
        time.sleep(3)
        proc_dead = _SERVER_PROCESS is not None and _SERVER_PROCESS.poll() is not None
        if not proc_dead and _server_alive():
            continue
        if _restart_server() and reload_callback:
            try:
                reload_callback()
            except Exception:
                pass


def run_native():
    ready = start_flask()
    if not ready:
        raise RuntimeError('Debug Panel Web 服务启动失败，请检查 runtime_outputs/gui_server.log')

    try:
        import gi
        gi.require_version('Gtk', '3.0')
        gi.require_version('WebKit2', '4.0')
        from gi.repository import Gtk, WebKit2, GLib

        win = Gtk.Window(title='Debug Panel')
        win.set_default_size(700, 800)
        win.set_position(Gtk.WindowPosition.CENTER)
        win.set_resizable(True)

        web = WebKit2.WebView()
        web.get_settings().set_enable_javascript(True)
        web.load_uri(_server_url())
        threading.Thread(
            target=_monitor_server,
            args=(lambda: GLib.idle_add(web.load_uri, _server_url()),),
            daemon=True,
            name='debug-panel-server-watchdog',
        ).start()

        win.add(web)
        win.connect('destroy', lambda *_args: (_cleanup_server(), Gtk.main_quit()))
        win.show_all()
        Gtk.main()
    except Exception as exc:
        print(f'窗口失败: {exc}')
        webbrowser.open(_server_url())
        try:
            while True:
                if _SERVER_PROCESS is not None and _SERVER_PROCESS.poll() is not None:
                    _restart_server()
                time.sleep(60)
        finally:
            if _STARTED_SERVER:
                _cleanup_server()
