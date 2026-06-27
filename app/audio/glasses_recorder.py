class GlassesRecorder:
    def __init__(self, bridge, log=None):
        self._bridge = bridge
        self._log = log or (lambda msg: None)

    def start(self):
        if not self._bridge or not self._bridge.enabled:
            return False
        if not self._bridge.start_recording():
            self._log('⚠️ 眼镜麦克风开始录音失败')
            return False
        return True

    def stop(self):
        if not self._bridge or not self._bridge.enabled:
            return None
        return self._bridge.stop_recording()
