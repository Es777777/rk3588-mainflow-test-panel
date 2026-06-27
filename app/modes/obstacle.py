from app.modes.base import BaseMode


class NavRecognitionMode(BaseMode):
    name = '导航识别模式'
    uses_rk3588_camera = True
    s2_press_log = '🎤 导航识别录音中...'
    s2_release_log = '📝 转写中...'
    release_log_prefix = '导航识别'
    release_too_short_message = '⏳ 录音时间过短，请重新按住按钮'
    release_allowed_intents = {'obstacle_query', 'chat', 'scene_explain'}
    release_handler_name = 'handle_obstacle_release'

    def __init__(self, state):
        super().__init__(state)
        self._last_restart_ts = 0.0

    def on_enter(self):
        self.state.add_log('🔄 导航识别模式启动')
        self.state.activate_obstacle_runtime('进入导航识别模式')
        self.state.add_log('🔇 导航识别模式提示音由眼镜固件处理')
        self.state.consume_pending_agent_action(self.name)

    def on_exit(self):
        if self.state.rk3588_obstacle and self.state.rk3588_obstacle.running:
            stopped = self.state.rk3588_obstacle.stop()
            self.state.add_log('🛑 已关闭 RK3588 避障进程' if stopped else '⚠️ RK3588 避障进程未能完全关闭')

    def on_s2_release(self):
        audio_path = self.state.recorder.stop()
        if not audio_path:
            self.state.add_log(self.release_too_short_message)
            self.state.speak_text('请按住说话')
            return

        text = self.state.transcribe_audio(audio_path, log_prefix=self.release_log_prefix)
        if not text:
            self.state.speak_text('没有听清')
            return

        self.state.set_stt_text(text)
        self.state.add_log(f'🗣️ 识别文本: {text}')
        self.state.execute_obstacle_reply({
            'intent': 'obstacle_query',
            'user_text': text,
        })

    def update(self):
        if not self.state.rk3588_obstacle:
            return
        current_mode = self.state.current_mode()
        if current_mode and current_mode.name != self.name:
            return
        if self.state.mode_transitioning():
            return
        if not self.state.rk3588_obstacle.running:
            current_mode = self.state.current_mode()
            if self.state.mode_transitioning() or (current_mode and current_mode.name != self.name):
                return
            now = __import__('time').time()
            if now - self._last_restart_ts >= 2.0:
                self._last_restart_ts = now
                self.state.add_log('🔁 避障运行时未在运行，正在尝试重新拉起')
                self.state.rk3588_obstacle.start()
            return
        payload = self.state.rk3588_obstacle.read_latest_payload()
        if payload:
            age_s = self.state.rk3588_obstacle.output_age_s()
            if age_s is not None and age_s >= 8.0:
                now = __import__('time').time()
                if now - self._last_restart_ts >= 3.0:
                    self._last_restart_ts = now
                    self.state.add_log(f'🔁 避障输出停更 {age_s:.1f}s，尝试重启运行时')
                    self.state.rk3588_obstacle.restart(f'输出停更 {age_s:.1f}s')
                    return
            self.state.apply_obstacle_payload(payload, speak=False)
            self._last_restart_ts = 0.0
            return
        age_s = self.state.rk3588_obstacle.output_age_s()
        if age_s is not None and age_s >= 8.0:
            now = __import__('time').time()
            if now - self._last_restart_ts >= 3.0:
                self._last_restart_ts = now
                self.state.add_log(f'🔁 避障向量超时 {age_s:.1f}s，尝试重启运行时')
                self.state.rk3588_obstacle.restart(f'输出停更 {age_s:.1f}s')
                return
        self.state.set_obstacle_payload({})
        self.state.set_obstacles([])
