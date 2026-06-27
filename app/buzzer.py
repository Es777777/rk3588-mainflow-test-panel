import shutil
import subprocess
import threading


class Buzzer:
    def __init__(self, config=None):
        self._config = config or {}

    def beep_once(self):
        threading.Thread(target=self._beep_once, daemon=True).start()

    def _beep_once(self):
        cmd = self._configured_command()
        if cmd:
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
                return
            except Exception:
                pass

        if shutil.which('beep'):
            try:
                subprocess.run(['beep', '-f', '1800', '-l', '120'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
                return
            except Exception:
                pass

        if shutil.which('play'):
            try:
                subprocess.run(
                    ['play', '-nq', '-t', 'alsa', 'synth', '0.12', 'sine', '1800'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
                return
            except Exception:
                pass

        if shutil.which('speaker-test'):
            try:
                subprocess.run(
                    ['speaker-test', '-t', 'sine', '-f', '1800', '-l', '1'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
                return
            except Exception:
                pass

        print('\a', end='', flush=True)

    def _configured_command(self):
        cmd = self._config.get('command')
        if isinstance(cmd, list) and cmd:
            return [str(part) for part in cmd]
        return None
