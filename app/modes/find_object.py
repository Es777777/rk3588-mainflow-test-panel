import os
import csv
from app.modes.base import BaseMode
from app.config import get_config


class FindObjectMode(BaseMode):
    name = '寻物模式'

    def __init__(self, state):
        super().__init__(state)
        self.objects = self._load_objects()
        self.target = None
        self.stt_text = ''
        self.detection_result = None

    def _load_objects(self):
        cfg = get_config()
        path = cfg.get('objects_file', 'app/data/objects.txt')
        if not os.path.isabs(path):
            base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            path = os.path.join(base, path)
        items = {}
        if os.path.exists(path):
            with open(path) as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        eng = row[0].strip()
                        aliases = [a.strip() for a in row[1:] if a.strip()]
                        items[eng] = aliases
        return items

    def extract_target(self, text):
        for eng, aliases in self.objects.items():
            if eng.lower() in text.lower():
                return eng
            for alias in aliases:
                if alias in text:
                    return eng
        return None

    def on_enter(self):
        self.target = None
        self.stt_text = ''
        self.detection_result = None
        if self.state.tts.available:
            self.state.tts.speak('当前为寻物模式')
        else:
            self.state.add_log('🔇 TTS 不可用')

    def on_s2_press(self):
        self.state.recorder.start()
        self.state.add_log('🎤 录音中...')

    def on_s2_release(self):
        audio_path = self.state.recorder.stop()
        if not audio_path:
            return
        self.state.add_log('📝 转写中...')
        text = self.state.stt.transcribe(audio_path)
        self.stt_text = text
        self.state.set_stt_text(text)
        if not text:
            if self.state.tts.available:
                self.state.tts.speak('请问要找什么')
            self.state.add_log('❓ 未识别到语音')
            return
        target = self.extract_target(text)
        if not target:
            if self.state.tts.available:
                self.state.tts.speak('暂不支持查找该物品')
            self.state.add_log(f'❌ 物品不在列表中: {text}')
            return
        self.target = target
        self.state.set_target(target)
        self.state.add_log(f'🔍 目标: {target}')
        if self.state.yolo and self.state.yolo.available and self.state.camera.available:
            frame = self.state.camera.get_frame_raw()
            if frame is not None:
                dets = self.state.yolo.detect(frame, target)
                matches = [d for d in dets if d['label'] == target]
                if matches:
                    d = matches[0]
                    self.detection_result = {**d, 'found': True, 'target': target}
                    self.state.set_detection(self.detection_result)
                    msg = f'已找到{target}'
                    self.state.add_log(f'✅ {msg}')
                    if self.state.tts.available:
                        self.state.tts.speak(msg)
                else:
                    self.detection_result = {'found': False, 'target': target}
                    self.state.set_detection(self.detection_result)
                    msg = f'未找到{target}'
                    self.state.add_log(f'❌ {msg}')
                    if self.state.tts.available:
                        self.state.tts.speak(msg)
