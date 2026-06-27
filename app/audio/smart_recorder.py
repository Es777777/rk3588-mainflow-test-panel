from app.audio.glasses_recorder import GlassesRecorder
from app.audio.recorder import Recorder


class SmartRecorder:
    def __init__(self, glasses_bridge=None, log=None):
        self._log = log or (lambda msg: None)
        self._glasses_bridge = glasses_bridge
        self._glasses_recorder = GlassesRecorder(glasses_bridge, log=log) if glasses_bridge else None
        self._local_recorder = Recorder()
        self._active_backend = None

    def start(self):
        if self._glasses_bridge and self._glasses_bridge.connected and self._glasses_recorder:
            if self._glasses_recorder.start():
                self._active_backend = 'glasses'
                return True
            self._log('⚠️ 眼镜录音不可用，回退到本地麦克风')

        started = bool(self._local_recorder.start())
        if started:
            self._active_backend = 'local'
        return started

    def stop(self):
        backend = self._active_backend
        self._active_backend = None
        if backend == 'glasses' and self._glasses_recorder:
            return self._glasses_recorder.stop()
        if backend == 'local':
            return self._local_recorder.stop()
        return None
