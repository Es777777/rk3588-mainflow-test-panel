class BaseMode:
    name = 'base'
    uses_local_camera = False
    uses_rk3588_camera = False
    s2_press_log = ''
    s2_release_log = ''
    release_log_prefix = ''
    release_too_short_message = ''
    release_allowed_intents = set()
    release_handler_name = ''
    release_empty_speak_text = ''
    release_unsupported_target_text = ''
    release_unsupported_target_log = ''
    release_allow_target_llm = False

    def __init__(self, state):
        self.state = state

    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def on_s2_press(self):
        started = bool(self.state.recorder.start())
        if started and self.s2_press_log:
            self.state.add_log(self.s2_press_log)
        elif not started:
            self.state.add_log('⚠️ 录音启动失败')
        return started

    def on_s2_release(self):
        audio_path = self.state.recorder.stop()
        if self.s2_release_log:
            self.state.add_log(self.s2_release_log)
        handler = getattr(self.state, self.release_handler_name, None) if self.release_handler_name else None
        if not handler:
            return
        self.state.handle_mode_agent_release(
            audio_path,
            log_prefix=self.release_log_prefix,
            too_short_message=self.release_too_short_message,
            allowed_intents=self.release_allowed_intents,
            mode_handler=handler,
            empty_speak_text=self.release_empty_speak_text,
            unsupported_target_text=self.release_unsupported_target_text,
            unsupported_target_log=self.release_unsupported_target_log,
            allow_target_llm=self.release_allow_target_llm,
        )

    def update(self):
        pass
