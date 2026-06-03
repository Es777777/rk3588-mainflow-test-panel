import threading
import time
import logging
from datetime import datetime

from app.config import load_config
from app.switches import SwitchManager
from app.device_check import run_all as check_all_devices
from app.audio.stt import STT
from app.audio.tts import TTS
from app.audio.recorder import Recorder
from app.vision.camera import Camera
from app.vision.yolo_detector import YOLODetector
from app.vision.intern_model import InternModel
from app.modes.find_object import FindObjectMode
from app.modes.obstacle import ObstacleMode
from app.modes.navigation import NavigationMode


class AppState:
    def __init__(self):
        self.logs = []
        self.mode_index = 0
        self.modes = []
        self.device_status = {}
        self.stt_text = ''
        self.target = ''
        self.detection = {}
        self.obstacles = []
        self.camera = Camera()
        self.yolo = YOLODetector()
        self.stt = STT()
        self.tts = TTS()
        self.intern = InternModel()
        self.recorder = Recorder()

    def set_mode(self, idx):
        if self.modes:
            self.modes[self.mode_index].on_exit()
        self.mode_index = idx % len(self.modes)
        self.modes[self.mode_index].on_enter()

    def cycle_mode(self):
        self.set_mode(self.mode_index + 1)

    def current_mode(self):
        return self.modes[self.mode_index] if self.modes else None

    def add_log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        entry = f'[{ts}] {msg}'
        self.logs.append(entry)
        logging.info(msg)

    def set_stt_text(self, text):
        self.stt_text = text

    def set_target(self, target):
        self.target = target

    def set_detection(self, det):
        self.detection = det

    def set_obstacles(self, obs):
        self.obstacles = obs


def create_app():
    load_config()

    state = AppState()

    mode_args = [state]
    state.modes = [
        FindObjectMode(state),
        ObstacleMode(state),
        NavigationMode(state),
    ]

    state.add_log('🚀 应用启动中...')

    state.camera.check()
    if state.camera.check():
        state.camera.start()
        state.add_log('📷 摄像头已初始化')

    state.stt.check()
    state.add_log('🎙️ STT 初始化完成' if state.stt._available else '⚠️ STT 未安装 (pip install openai-whisper)')

    state.tts.check()
    state.add_log('🔊 TTS 初始化完成' if state.tts._available else '⚠️ TTS 未安装 (apt install espeak)')

    state.intern.check()
    state.add_log('🧠 Intern 模型' + ('已加载' if state.intern._available else '未加载 (需安装 transformers)'))

    state.yolo.check()
    state.add_log('👁️ YOLO' + ('已就绪' if state.yolo._available else '未安装 (pip install ultralytics)'))

    state.add_log('🔍 设备自检中...')
    state.device_status = check_all_devices()
    for name, info in state.device_status['checks'].items():
        icon = '✅' if info['ok'] else '❌'
        state.add_log(f'{icon} {name}: {info["detail"]}')

    state.set_mode(0)

    def mode_update_loop():
        while True:
            try:
                mode = state.current_mode()
                if mode:
                    mode.update()
            except Exception:
                pass
            time.sleep(0.1)

    threading.Thread(target=mode_update_loop, daemon=True).start()

    state.add_log('✅ 应用启动完成')
    return state
