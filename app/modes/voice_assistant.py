import os
from app.modes.base import BaseMode


class VoiceAssistantMode(BaseMode):
    name = '语音助手'
    uses_local_camera = True

    def on_enter(self):
        self.state.add_log('🗣️ 语音助手模式启动')
        self.state.pause_realtime_runtimes('进入语音助手模式')
        self.state.add_log('🔇 语音助手模式提示音由眼镜固件处理')

    def on_s2_press(self):
        self.state.add_log('🎤 语音助手录音中...')
        return bool(self.state.recorder.start())

    def on_s2_release(self):
        audio_path = self.state.recorder.stop()
        if not audio_path:
            self.state.add_log('⏳ 录音时间过短')
            self.state.speak_text('请按住说话')
            return

        text = self.state.transcribe_audio(audio_path, log_prefix='语音助手')
        if not text:
            self.state.speak_text('没有听清')
            return

        self.state.add_log(f'🗣️ 识别文字: {text}')

        image_path = self.state.capture_agent_snapshot(crop_left=True)
        if not image_path:
            self.state.speak_text('没有获取到画面')
            return

        reply = self.state.analyze_voice_assistant_image(text, image_path)
        self.state.speak_text(reply or '分析失败')
