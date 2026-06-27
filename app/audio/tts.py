import os
import subprocess
import tempfile
import threading
import time


class TTS:
    def __init__(self):
        self._engine = None
        self._available = False

    def check(self):
        if self._try_edge():
            self._engine = 'edge'
            self._available = True
            return True
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

    def _try_edge(self):
        try:
            import edge_tts
            return True
        except ImportError:
            return False

    def speak(self, text):
        if not self._available:
            return
        threading.Thread(target=self._speak_async, args=(text,), daemon=True).start()

    @property
    def available(self):
        return self._available

    @staticmethod
    def _ensure_volume():
        subprocess.run(['amixer', '-c', '4', 'sset', 'Headphone', '3', 'unmute'], stderr=subprocess.DEVNULL)
        subprocess.run(['amixer', '-c', '4', 'sset', 'Headphone Mixer', '11'], stderr=subprocess.DEVNULL)
        subprocess.run(['amixer', '-c', '4', 'sset', 'DAC', '192'], stderr=subprocess.DEVNULL)
        subprocess.run(['amixer', '-c', '4', 'sset', 'Speaker', 'on'], stderr=subprocess.DEVNULL)
        subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', '100%'], stderr=subprocess.DEVNULL)
        subprocess.run(['pactl', 'set-sink-mute', '@DEFAULT_SINK@', '0'], stderr=subprocess.DEVNULL)

    def _speak_async(self, text):
        try:
            self._ensure_volume()
            if self._engine == 'edge':
                import edge_tts
                import asyncio
                tmp = tempfile.mktemp(suffix='.mp3')
                asyncio.run(edge_tts.Communicate(text, 'zh-CN-XiaoxiaoNeural').save(tmp))
                subprocess.run(['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', tmp],
                               timeout=60, stderr=subprocess.DEVNULL)
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
            elif self._engine == 'espeak':
                subprocess.run(['espeak', text], timeout=30)
            elif self._engine == 'festival':
                subprocess.run(['festival', '--tts'], input=text.encode(), timeout=30)
        except Exception:
            pass
