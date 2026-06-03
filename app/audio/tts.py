import subprocess
import threading


class TTS:
    def __init__(self):
        self._engine = None
        self._available = False

    def check(self):
        for cmd in ['espeak', 'festival', 'say']:
            if self._find(cmd):
                self._engine = cmd
                self._available = True
                return True
        self._available = False
        return False

    def _find(self, cmd):
        import shutil
        return shutil.which(cmd) is not None

    def speak(self, text):
        if not self._available:
            return
        threading.Thread(target=self._speak_async, args=(text,), daemon=True).start()

    @property
    def available(self):
        return self._available

    def _speak_async(self, text):
        try:
            if self._engine == 'espeak':
                subprocess.run(['espeak', text], timeout=30)
            elif self._engine == 'festival':
                subprocess.run(['festival', '--tts'], input=text.encode(), timeout=30)
        except Exception:
            pass
