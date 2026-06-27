import os
import subprocess
import tempfile
import threading
import time
import wave
from app.config import get_config


class Recorder:
    def __init__(self):
        self._recording = False
        self._frames = []
        self._lock = threading.Lock()
        self._thread = None
        self._tmp_path = None
        self._proc = None
        self._start_time = 0

    def start(self):
        if self._recording:
            return False
        self._recording = True
        self._frames = []
        self._tmp_path = None
        self._proc = None
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()
        return True

    def _record(self):
        try:
            import pyaudio
            cfg = get_config()
            rate = cfg['audio']['sample_rate']
            channels = cfg['audio']['channels']
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=channels,
                            rate=rate, input=True, frames_per_buffer=1024)
            frames = []
            while self._recording:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
        except Exception:
            self._record_arecord()
            return
        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception:
            pass
        with self._lock:
            self._frames = frames

    def _record_arecord(self):
        cfg = get_config()
        rate = cfg['audio']['sample_rate']
        channels = cfg['audio']['channels']
        fd, self._tmp_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        cmd = [
            'arecord', '-D', 'plughw:4,0',
            '-f', 'S16_LE', '-c', str(channels), '-r', str(rate),
            self._tmp_path
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while self._recording:
            try:
                self._proc.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                continue
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

    def stop(self):
        self._recording = False
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=5)
        if self._tmp_path and os.path.isfile(self._tmp_path):
            sz = os.path.getsize(self._tmp_path)
            if sz > 44:
                result = self._tmp_path
                self._tmp_path = None
                return result
            os.unlink(self._tmp_path)
        return self._save_temp()

    def _save_temp(self):
        with self._lock:
            frames = list(self._frames)
            self._frames.clear()
        if not frames:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        with wave.open(tmp.name, 'wb') as wf:
            cfg = get_config()
            wf.setnchannels(cfg['audio']['channels'])
            wf.setsampwidth(2)
            wf.setframerate(cfg['audio']['sample_rate'])
            wf.writeframes(b''.join(frames))
        return tmp.name
