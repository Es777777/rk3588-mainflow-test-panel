from app.modes.base import BaseMode


class ObstacleMode(BaseMode):
    name = '避障模式'

    def on_enter(self):
        self.state.add_log('🔄 避障模式启动')
        if self.state.tts.available:
            self.state.tts.speak('当前为避障模式')

    def update(self):
        if not self.state.yolo or not self.state.yolo.available:
            return
        if not self.state.camera or not self.state.camera.available:
            return
        frame = self.state.camera.get_frame_raw()
        if frame is None:
            return
        dets = self.state.yolo.detect(frame)
        obstacles = [d for d in dets if d['label'] in ('person', 'chair', 'table', 'dog', 'cat', 'bicycle', 'car', 'motorcycle', 'bus', 'truck')]
        if obstacles:
            self.state.set_obstacles(obstacles)
