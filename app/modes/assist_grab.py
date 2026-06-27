from app.modes.base import BaseMode


class AssistGrabMode(BaseMode):
    name = '辅助抓取模式'
    uses_rk3588_camera = True
    s2_press_log = '🎤 辅助抓取目标录音中...'
    release_log_prefix = '辅助抓取'
    release_too_short_message = '⏳ 录音时间过短，请重新按住按钮'
    release_allowed_intents = {'assist_grab', 'scene_explain', 'chat'}
    release_handler_name = 'handle_assist_grab_release'
    release_empty_speak_text = '未提取出目标物体'
    release_unsupported_target_text = '未提取出目标物体'
    release_unsupported_target_log = '❌ 未从语音中提取出 YOLO 目标物体'
    release_allow_target_llm = True

    def __init__(self, state):
        super().__init__(state)
        self._prompted_waiting = False
        self._startup_announced = False
        self._last_restart_ts = 0.0

    def on_enter(self):
        self._prompted_waiting = False
        self._startup_announced = False
        self._ensure_vector_runtime()
        self.state.add_log('🤖 辅助抓取模式启动')
        self.state.consume_pending_agent_action(self.name)

    def on_exit(self):
        self._startup_announced = False
        if self.state.rk3588 and self.state.rk3588.running:
            stopped = self.state.rk3588.stop()
            self.state.add_log('🛑 已关闭辅助抓取 RK3588 向量进程' if stopped else '⚠️ 辅助抓取 RK3588 向量进程未能完全关闭')

    def on_s2_release(self):
        audio_path = self.state.recorder.stop()
        if not audio_path:
            self.state.add_log(self.release_too_short_message)
            self.state.speak_text(self.release_empty_speak_text or '请按住说话')
            return

        text = self.state.transcribe_audio(audio_path, log_prefix=self.release_log_prefix)
        if not text:
            self.state.speak_text(self.release_empty_speak_text or '没有听清')
            return

        self.state.set_stt_text(text)
        self.state.add_log(f'🗣️ 识别文本: {text}')
        target = self.state.resolve_target_object(text=text, allow_llm=self.release_allow_target_llm)
        if not target:
            log_message = self.release_unsupported_target_log or '❌ 未从语音中提取出 YOLO 目标物体'
            self.state.add_log(f'{log_message}: {text}')
            self.state.speak_text(self.release_unsupported_target_text or '未提取出目标物体')
            return

        self.state.execute_assist_grab_target(target, {
            'user_text': text,
            'speak_text': f'开始辅助抓取{self.state.display_target_name(target) or target}',
        })

    def update(self):
        if not self.state.rk3588:
            return
        current_mode = self.state.current_mode()
        if current_mode and current_mode.name != self.name:
            return
        if self.state.mode_transitioning():
            return
        if self.state.npu_exclusive_active():
            return
        if not self.state.rk3588.running:
            current_mode = self.state.current_mode()
            if self.state.mode_transitioning() or (current_mode and current_mode.name != self.name):
                return
            now = __import__('time').time()
            if now - self._last_restart_ts >= 2.0:
                self._last_restart_ts = now
                self.state.add_log('🔁 辅助抓取运行时未在运行，正在尝试重新拉起')
                self.state.rk3588.start()
            return
        payload = self.state.rk3588.read_latest_payload()
        if payload:
            age_s = self.state.rk3588.output_age_s()
            if age_s is not None and age_s >= 8.0:
                now = __import__('time').time()
                if now - self._last_restart_ts >= 3.0:
                    self._last_restart_ts = now
                    self.state.add_log(f'🔁 辅助抓取输出停更 {age_s:.1f}s，尝试重启运行时')
                    self.state.rk3588.restart(f'输出停更 {age_s:.1f}s')
                    return
            self.state.set_vector_payload(payload)
            self._prompted_waiting = False
            self._last_restart_ts = 0.0
            if self.state.rk3588.ready and not self._startup_announced:
                self._startup_announced = True
                self.state.add_log('🔇 辅助抓取模式提示音由眼镜固件处理')
        else:
            age_s = self.state.rk3588.output_age_s()
            if age_s is not None and age_s >= 8.0:
                now = __import__('time').time()
                if now - self._last_restart_ts >= 3.0:
                    self._last_restart_ts = now
                    self.state.add_log(f'🔁 辅助抓取向量超时 {age_s:.1f}s，尝试重启运行时')
                    self.state.rk3588.restart(f'输出停更 {age_s:.1f}s')
                    return
            if not self.state.rk3588.ready and not self._prompted_waiting:
                self._prompted_waiting = True
                self.state.add_log('⏳ 等待 RK3588 本地模型输出向量...')
            elif self.state.rk3588.ready and not self._prompted_waiting:
                self._prompted_waiting = True
                self.state.add_log('⏳ 等待 RK3588 本地模型输出向量...')

    def _ensure_vector_runtime(self):
        if not self.state.rk3588:
            return
        if not self.state.rk3588.running:
            self.state.activate_vector_runtime('进入辅助抓取模式')
        if self.state._resetting_runtime:
            return
        default_target = str(self.state.config.get('rk3588_runtime', {}).get('default_target_class', '')).strip()
        if not self.state.target and default_target and self.state.rk3588.set_target_class(default_target):
            self.state.set_target(default_target)
            self.state.update_task_context({
                'intent': 'assist_grab',
                'mode_name': self.name,
                'target_object': default_target,
            })
            self.state.add_log(f'🧭 辅助抓取默认目标: {default_target}')
