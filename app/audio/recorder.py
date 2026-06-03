import tempfile
import wave
import threading
from app.config import get_config


class Recorder:
    def __init__(self):
        self._recording = False
        self._frames = []
        self._thread = None

    def start(self):
        if self._recording:
            return
        self._recording = True
        self._frames = []
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()

    def _record(self):
        try:
            import pyaudio
            cfg = get_config()
            rate = cfg['audio']['sample_rate']
            channels = cfg['audio']['channels']
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=channels,
                            rate=rate, input=True, frames_per_buffer=1024)
            while self._recording:
                data = stream.read(1024, exception_on_overflow=False)
                self._frames.append(data)
            stream.stop_stream()
            stream.close()
            p.terminate()
        except ImportError:
            pass

    def stop(self):
        self._recording = False
        if self._thread:
            self._thread.join(timeout=2)
        return self._save_temp()

    def _save_temp(self):
        if not self._frames:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        with wave.open(tmp.name, 'wb') as wf:
            cfg = get_config()
            wf.setnchannels(cfg['audio']['channels'])
            wf.setsampwidth(2)
            wf.setframerate(cfg['audio']['sample_rate'])
            wf.writeframes(b''.join(self._frames))
        return tmp.name
