import json
import os
import shutil
import subprocess
import threading
import time

from app.config import resolve_project_path


class RK3588ObstacleBridge:
    def __init__(self, config=None, log=None):
        self._config = config or {}
        self._log = log or (lambda msg: None)
        self._proc = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._ready = False
        self._last_payload = {}
        self._last_payload_ts = 0.0
        self._last_payload_mtime = 0.0
        self._last_error = ''
        self._lock = threading.Lock()
        self._proc_lock = threading.RLock()

    @property
    def enabled(self):
        return bool(self._config.get('enabled', False))

    @property
    def ready(self):
        return self._ready

    @property
    def running(self):
        return self._proc is not None and self._proc.poll() is None

    @property
    def rolling_output_path(self):
        return self._abs_path(self._config.get('rolling_output_path', ''))

    @property
    def history_output_path(self):
        return self._abs_path(self._config.get('history_output_path', ''))

    @property
    def csv_output_path(self):
        return self._abs_path(self._config.get('csv_output_path', ''))

    @property
    def preview_output_path(self):
        return self._abs_path(self._config.get('preview_output_path', ''))

    @property
    def last_error(self):
        return self._last_error

    def check(self):
        script_path = self._abs_path(self._config.get('script_path', ''))
        config_path = self._abs_path(self._config.get('config_path', ''))
        python_path = self._resolve_executable(self._config.get('python_path', ''))
        return bool(
            script_path
            and os.path.exists(script_path)
            and config_path
            and os.path.exists(config_path)
            and python_path
        )

    def start(self):
        with self._proc_lock:
            if not self.enabled or self.running:
                return
            if not self.check():
                self._last_error = 'RK3588 避障运行时路径不完整'
                self._log(f'⚠️ {self._last_error}')
                return
            self._ready = False
            self._last_error = ''
            self._clear_cached_payload()
            self._prepare_output_paths()

            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            cmd = [
                self._resolve_executable(self._config.get('python_path', '')),
                self._abs_path(self._config.get('script_path', '')),
                '--config',
                self._abs_path(self._config.get('config_path', '')),
            ]
            cwd = self._abs_path(self._config.get('working_dir', ''))
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=cwd or None,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                self._stdout_thread = threading.Thread(target=self._consume_pipe, args=(self._proc.stdout, 'OBSTACLE'), daemon=True)
                self._stderr_thread = threading.Thread(target=self._consume_pipe, args=(self._proc.stderr, 'OBSTACLE ERR'), daemon=True)
                self._stdout_thread.start()
                self._stderr_thread.start()
                threading.Thread(target=self._watch_ready, daemon=True).start()
                self._log('🚀 RK3588 避障运行时已启动')
            except Exception as exc:
                self._last_error = str(exc)
                self._log(f'❌ RK3588 避障运行时启动失败: {exc}')

    def stop(self):
        with self._proc_lock:
            proc = self._proc
            if not proc:
                return True
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self._last_error = 'RK3588 避障运行时无法终止，可能卡在内核/摄像头驱动'
                        self._log(f'⚠️ {self._last_error}')
                        return False
            self._proc = None
            self._ready = False
            self._clear_cached_payload()
            return True

    def read_latest_payload(self):
        payload, mtime = self._read_payload_from(self.rolling_output_path)
        if not payload:
            payload, mtime = self._read_payload_from(self.history_output_path)
        if not payload:
            return dict(self._last_payload)
        with self._lock:
            self._last_payload = payload
            if mtime and mtime != self._last_payload_mtime:
                self._last_payload_mtime = mtime
                self._last_payload_ts = time.time()
        return dict(payload)

    @property
    def last_payload_ts(self):
        with self._lock:
            return self._last_payload_ts

    def payload_age_s(self):
        with self._lock:
            if not self._last_payload_ts:
                return None
            return max(0.0, time.time() - self._last_payload_ts)

    def output_age_s(self):
        mtimes = []
        for path in (self.rolling_output_path, self.preview_output_path):
            if path and os.path.exists(path):
                try:
                    mtimes.append(os.path.getmtime(path))
                except OSError:
                    pass
        if not mtimes:
            return None
        return max(0.0, time.time() - max(mtimes))

    def restart(self, reason=''):
        with self._proc_lock:
            if reason:
                self._last_error = reason
                self._log(f'🔁 RK3588 避障运行时重启: {reason}')
            if not self.stop():
                return False
            self.start()
            return True

    def _watch_ready(self):
        deadline = time.time() + float(self._config.get('ready_timeout_s', 45))
        while self.running and time.time() < deadline:
            payload = self.read_latest_payload()
            if payload:
                self._ready = True
                self._log('✅ RK3588 避障输出已就绪')
                return
            time.sleep(0.5)
        if not self._ready and not self._last_error and not self.running:
            self._last_error = 'RK3588 避障运行时已退出'

    def _consume_pipe(self, pipe, prefix):
        if pipe is None:
            return
        try:
            for line in pipe:
                text = line.strip()
                if text:
                    self._log(f'[{prefix}] {text}')
        except Exception:
            pass

    def _abs_path(self, path):
        if not path:
            return ''
        return resolve_project_path(path)

    def _resolve_executable(self, path):
        if not path:
            return ''
        value = os.path.expanduser(str(path))
        if os.path.isabs(value) or os.sep in value or (os.altsep and os.altsep in value):
            resolved = resolve_project_path(value)
            return resolved if os.path.exists(resolved) else ''
        return shutil.which(value) or value

    def _prepare_output_paths(self):
        for path in (self.rolling_output_path, self.history_output_path, self.csv_output_path, self.preview_output_path):
            if not path:
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    def _clear_cached_payload(self):
        with self._lock:
            self._last_payload = {}
            self._last_payload_ts = 0.0
            self._last_payload_mtime = 0.0

    def _read_payload_from(self, path):
        if not path or not os.path.exists(path):
            return None, 0.0
        try:
            mtime = os.path.getmtime(path)
            with open(path, 'rb') as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                seek_pos = max(0, size - 2048)
                handle.seek(seek_pos)
                chunk = handle.read().decode('utf-8')
            lines = chunk.strip().split('\n')
            line = lines[-1].strip() if lines else ''
            if not line:
                return None, 0.0
            return json.loads(line), mtime
        except Exception:
            return None, 0.0
