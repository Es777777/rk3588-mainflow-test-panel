import os
import gc
import subprocess
import threading
import time
import logging
import csv
import cv2
import shutil
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from app.config import load_config, resolve_project_path
from app.device_check import run_all as check_all_devices, set_camera_state
from app.audio.stt import STT
from app.audio.tts import TTS
from app.audio.smart_recorder import SmartRecorder
from app.audio.rkllama_client import RkllamaClient
from app.cloud_vision_client import CloudVisionClient
from app.glasses_bridge import GlassesBridge
from app.wristband_bridge import WristbandBridge
from app.rk3588_bridge import RK3588VectorBridge
from app.rk3588_obstacle_bridge import RK3588ObstacleBridge
from app.vision.camera import Camera
from app.vision.yolo_detector import YOLODetector
from app.vision.intern_model import InternModel
from app.agent.local_multimodal_agent import LocalMultimodalAgent
from app.modes.assist_grab import AssistGrabMode
from app.modes.obstacle import NavRecognitionMode
from app.modes.voice_assistant import VoiceAssistantMode
from app.text_normalize import to_simplified


class AppState:
    _OBSTACLE_LABELS = {
        'bench': '长椅',
        'bicycle': '自行车',
        'bus': '公交车',
        'bus_stop': '公交站',
        'cane': '手杖',
        'car': '汽车',
        'curb': '路沿',
        'dog': '狗',
        'fire_hydrant': '消防栓',
        'motorcycle': '摩托车',
        'person': '行人',
        'pole': '立杆',
        'spherical_roadblock': '球形路障',
        'stairs': '台阶',
        'stop_sign': '停止标志',
        'street_light': '路灯',
        'traffic_light': '红绿灯',
        'train': '火车',
        'tree': '树',
        'truck': '卡车',
        'warning_column': '警示柱',
        'waste_container': '垃圾桶',
        'blind_road': '盲道',
        'bollard': '隔离柱',
        'traffic_cone': '路锥',
    }

    def __init__(self):
        self.config = load_config()
        self.logs = []
        self._events_log_path = Path(resolve_project_path('runtime_outputs/app_events.log'))
        self._events_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._startup_ts = time.time()
        self._accept_glasses_buttons = False
        self.mode_index = 0
        self.modes = []
        self.device_status = {}
        self._lock = threading.RLock()
        self.stt_text = ''
        self.target = ''
        self.detection = {}
        self.obstacles = []
        self.obstacle_payload = {}
        self.vector_payload = {}
        self.agent_result = {}
        self.pending_agent_action = None
        self.task_context = {}
        self.objects = self._load_objects()
        self.waiting_for_glasses_button = False
        self.camera = Camera(log=self.add_log)
        self.yolo = YOLODetector()
        self.stt = STT()
        self.tts = TTS()
        self.rkllama = RkllamaClient(self.config.get('rkllama', {}), log=self.add_log)
        self.intern = InternModel(self.rkllama, log=self.add_log)
        self.cloud_vision = CloudVisionClient(self.config.get('cloud_vision', {}), log=self.add_log)
        self.agent = LocalMultimodalAgent(self.rkllama, log=self.add_log)
        self.glasses = GlassesBridge(self.config.get('glasses_sdk', {}), log=self.add_log)
        self.wristband = WristbandBridge(self.config.get('wristband', {}), log=self.add_log)
        self.recorder = SmartRecorder(self.glasses, log=self.add_log)
        self.rk3588 = RK3588VectorBridge(
            self.config.get('rk3588_runtime', {}),
            log=self.add_log,
            on_ready=self.on_rk3588_ready,
        )
        self.rk3588_obstacle = RK3588ObstacleBridge(
            self.config.get('rk3588_obstacle_runtime', {}),
            log=self.add_log,
        )
        self.glasses.add_event_handler(self._handle_glasses_event)
        self._camera_started_by_mode = False
        self._mode_name_to_index = {}
        self._agent_visual_source = ''
        self._agent_visual_summary = ''
        self._agent_visual_attempted = False
        self._mode_entered = False
        self._resetting_runtime = False
        self._s2_active = False
        self._s2_pressed_mode_index = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='app')
        self._cleanup_running = False
        self._cleanup_thread = None
        self._last_obstacle_speak_ts = 0.0
        self._last_obstacle_speech_key = ''
        self._rkllama_call_lock = threading.RLock()
        self._speech_async_lock = threading.Lock()
        self._speech_async_busy = False
        self._speech_async_last_done_ts = 0.0
        self._npu_exclusive_task = ''
        self._last_mode_event_ts = 0.0
        self._last_talk_event_ts = 0.0
        self._mode_event_min_interval_s = 0.8
        self._ignore_mode_after_talk_s = 1.5
        self._mode_transitioning = False
        npu_policy = self.config.get('npu_policy', {}) or {}
        self._npu_pause_drain_s = float(npu_policy.get('pause_drain_s', 0.6) or 0.6)
        self._npu_resume_delay_s = float(npu_policy.get('resume_delay_s', 0.3) or 0.3)
        self._npu_stop_timeout_s = float(npu_policy.get('stop_timeout_s', 3.0) or 3.0)
        self._npu_policy_logged = False

    @property
    def s2_active(self):
        with self._lock:
            return bool(self._s2_active)

    def _load_objects(self):
        path = self.config.get('objects_file', 'app/data/objects.txt')
        if not os.path.isabs(path):
            base = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(base, path)
        items = {}
        if os.path.exists(path):
            with open(path, encoding='utf-8') as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if row:
                        eng = row[0].strip()
                        aliases = [a.strip() for a in row[1:] if a.strip()]
                        items[eng] = aliases
        return items

    def set_mode(self, idx):
        if not self.modes:
            return
        with self._lock:
            if self._mode_transitioning:
                self.add_log('⏭️ 模式切换进行中，忽略重复切换')
                return
            self._mode_transitioning = True
            old_index = self.mode_index
            new_index = idx % len(self.modes)
        try:
            if self.modes and self._mode_entered:
                self.modes[old_index].on_exit()
            with self._lock:
                self.mode_index = new_index
                self._mode_name_to_index = {mode.name: i for i, mode in enumerate(self.modes)}
            self._sync_mode_resources(self.modes[new_index])
            self.modes[new_index].on_enter()
            self._mode_entered = True
            self._update_wristband_packet()
        finally:
            with self._lock:
                self._mode_transitioning = False

    def cycle_mode(self):
        if not self.modes:
            return
        if self._s2_active:
            self.add_log('⏭️ 录音进行中，忽略模式切换')
            return
        self.set_mode((self.mode_index + 1) % len(self.modes))

    def current_mode(self):
        with self._lock:
            return self.modes[self.mode_index] if self.modes else None

    def mode_names(self):
        with self._lock:
            return [mode.name for mode in self.modes]

    def set_mode_by_name(self, mode_name):
        if not mode_name:
            return False
        if not self._mode_name_to_index:
            self._mode_name_to_index = {mode.name: i for i, mode in enumerate(self.modes)}
        idx = self._mode_name_to_index.get(mode_name)
        if idx is None:
            return False
        if idx == self.mode_index:
            return True
        self.set_mode(idx)
        return True

    def _sync_mode_resources(self, mode):
        if mode is None:
            return
        wants_local = bool(getattr(mode, 'uses_local_camera', False))
        if wants_local:
            if not self._camera_started_by_mode:
                started = self.camera.start()
                self._camera_started_by_mode = bool(started)
                source = self.camera.active_source()
                if started:
                    self.add_log(f'📷 已为当前模式启用本地相机: {source or "初始化中"}')
                else:
                    self.add_log('⚠️ 当前模式需要本地相机，但启动失败')
        else:
            if self._camera_started_by_mode:
                self.camera.stop()
                self._camera_started_by_mode = False
                self.add_log('📷 已为 RK3588 模式释放本地相机')

    def add_log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        entry = f'[{ts}] {msg}'
        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > 500:
                self.logs = self.logs[-500:]
        logging.info(msg)
        try:
            with self._events_log_path.open('a', encoding='utf-8') as handle:
                handle.write(entry + '\n')
        except Exception:
            pass

    def speak_text(self, text):
        text = str(text or '').strip()
        if not text:
            return False
        try:
            if self.rkllama and self.glasses and self.glasses.connected:
                reply_dir = self.config.get('rkllama', {}).get('reply_dir', 'runtime_outputs/voice_targeting')
                spoke = self.run_with_rkllama(
                    '语音播报',
                    lambda: (
                        self.rkllama.ensure_available()
                        and self.rkllama.speak_via_glasses(self.glasses, text, reply_dir)
                    ),
                    pause_accelerators=False,
                )
                if spoke:
                    self.add_log(f'🔊 语音已播报: {text}')
                    return True
                self.add_log('⚠️ 眼镜播报失败，切换系统 TTS')
        except Exception as exc:
            self.add_log(f'⚠️ 眼镜语音播报失败，回退到系统 TTS: {exc}')
        if self.tts and self.tts.available:
            self.tts.speak(text)
            self.add_log(f'🔊 系统 TTS 已发起: {text}')
            return True
        return False

    def speak_text_async(self, text, reason=''):
        text = str(text or '').strip()
        if not text:
            return False
        with self._speech_async_lock:
            if self._speech_async_busy:
                self.add_log(f'⏭️ 语音播报忙，跳过{reason or "播报"}: {text}')
                return False
            self._speech_async_busy = True

        def _run():
            try:
                self.speak_text(text)
            finally:
                with self._speech_async_lock:
                    self._speech_async_busy = False
                    self._speech_async_last_done_ts = time.time()

        threading.Thread(target=_run, name='speech-async', daemon=True).start()
        return True

    def set_stt_text(self, text):
        with self._lock:
            self.stt_text = to_simplified(text)

    def set_target(self, target):
        with self._lock:
            self.target = target

    def set_detection(self, det):
        with self._lock:
            self.detection = det

    def clear_object_focus_state(self):
        with self._lock:
            self.target = ''
            self.detection = {}

    def set_obstacles(self, obs):
        with self._lock:
            self.obstacles = obs

    def set_obstacle_payload(self, payload):
        with self._lock:
            self.obstacle_payload = payload or {}
        self._update_wristband_packet()

    def set_vector_payload(self, payload):
        with self._lock:
            self.vector_payload = payload or {}
        self._update_wristband_packet()

    def set_agent_result(self, payload):
        with self._lock:
            self.agent_result = payload or {}

    def set_pending_agent_action(self, payload):
        with self._lock:
            self.pending_agent_action = payload or None

    def _round_packet_float(self, value, default=0.0):
        try:
            return round(float(value), 6)
        except Exception:
            return round(float(default), 6)

    def _build_grasp_wristband_packet(self, payload):
        payload = payload or {}
        status = str(payload.get('status') or 'no_target')
        vector_active = bool(payload.get('vector_active')) and status == 'tracking'
        distance_default = 999.0
        if status == 'stop':
            distance_default = 0.03
        if not vector_active:
            return {
                'timestamp_s': round(time.time(), 6),
                'frame_index': int(payload.get('frame_index', 0) or 0),
                'status': status,
                'vector_active': False,
                'mode': 'grasp',
                'vector_x_m': 0.0,
                'vector_y_m': 0.0,
                'vector_z_m': 0.0,
                'distance_m': self._round_packet_float(payload.get('distance_m'), default=distance_default),
            }
        return {
            'timestamp_s': round(time.time(), 6),
            'frame_index': int(payload.get('frame_index', 0) or 0),
            'status': status,
            'vector_active': True,
            'mode': 'grasp',
            'vector_x_m': self._round_packet_float(payload.get('vector_x_m')),
            'vector_y_m': self._round_packet_float(payload.get('vector_y_m')),
            'vector_z_m': self._round_packet_float(payload.get('vector_z_m')),
            'distance_m': self._round_packet_float(payload.get('distance_m'), default=distance_default),
        }

    def _build_avoid_wristband_packet(self, payload):
        payload = payload or {}
        obstacle_name = str(payload.get('obstacle_class_name') or '').strip()
        vector_x = payload.get('vector_x_m')
        vector_z = payload.get('vector_z_m')
        distance_m = payload.get('distance_m')
        has_vector = obstacle_name and vector_x is not None and vector_z is not None
        if not has_vector:
            return {
                'timestamp_s': round(time.time(), 6),
                'frame_index': int(payload.get('frame_index', 0) or 0),
                'status': 'no_obstacle',
                'vector_active': False,
                'mode': 'avoid',
                'vector_x_m': 0.0,
                'vector_y_m': 0.0,
                'vector_z_m': 0.0,
                'distance_m': self._round_packet_float(distance_m, default=999.0),
            }
        return {
            'timestamp_s': round(time.time(), 6),
            'frame_index': int(payload.get('frame_index', 0) or 0),
            'status': 'tracking',
            'vector_active': True,
            'mode': 'avoid',
            'vector_x_m': self._round_packet_float(vector_x),
            'vector_y_m': 0.0,
            'vector_z_m': self._round_packet_float(vector_z),
            'distance_m': self._round_packet_float(distance_m, default=999.0),
        }

    def _update_wristband_packet(self):
        if not self.wristband or not self.wristband.enabled:
            return
        mode = self.current_mode()
        mode_name = mode.name if mode else ''
        with self._lock:
            vector_payload = dict(self.vector_payload or {})
            obstacle_payload = dict(self.obstacle_payload or {})
        packet = None
        if mode_name == '辅助抓取模式':
            packet = self._build_grasp_wristband_packet(vector_payload)
        elif mode_name == '导航识别模式':
            packet = self._build_avoid_wristband_packet(obstacle_payload)
        self.wristband.update_packet(packet)

    def update_task_context(self, updates=None):
        if not updates:
            with self._lock:
                return dict(self.task_context)
        with self._lock:
            merged = dict(self.task_context or {})
            for key, value in (updates or {}).items():
                if value is None:
                    continue
                if value == '':
                    merged.pop(key, None)
                    continue
                merged[key] = value
            self.task_context = merged
            return dict(self.task_context)

    def clear_task_context(self):
        with self._lock:
            self.task_context = {}

    def consume_pending_agent_action(self, mode_name=None):
        with self._lock:
            action = self.pending_agent_action
            if not action:
                return None
            target_mode = action.get('mode_name')
            current_mode = self.current_mode()
            current_name = mode_name or (current_mode.name if current_mode else None)
            if target_mode and current_name and target_mode != current_name:
                return None
            self.pending_agent_action = None
        handled = self.execute_pending_agent_action(action, current_mode=current_mode)
        if handled:
            self.add_log(f'✅ 已执行 Agent 接棒任务: {action.get("intent") or "unknown"}')
        else:
            self.add_log(f'⚠️ Agent 接棒任务未完成执行: {action.get("intent") or "unknown"}')
        if not handled and action.get('speak_text'):
            if self.speak_text(action.get('speak_text')):
                self.add_log('🔊 已使用通用播报兜底输出 Agent 回复')
        return action

    def execute_pending_agent_action(self, action, current_mode=None):
        action = action or {}
        mode = current_mode or self.current_mode()
        mode_name = mode.name if mode else ''
        intent = action.get('intent') or ''
        if intent in {'scene_explain', 'chat'}:
            outcome = self.execute_general_reply(action, default_intent='chat')
            return bool(outcome.get('ok'))
        if mode_name == '辅助抓取模式':
            target = action.get('target_object')
            if not target:
                return False
            self.add_log(f'🎯 Agent 接棒辅助抓取目标: {target}')
            outcome = self.execute_assist_grab_target(target, {
                'speak_text': action.get('speak_text') or f'开始辅助抓取{target}',
                'scene_summary': action.get('scene_summary') or '',
                'user_text': action.get('user_text') or '',
            })
            return bool(outcome.get('ok'))
        if mode_name == '导航识别模式':
            outcome = self.execute_obstacle_reply({
                'intent': intent or 'obstacle_query',
                'speak_text': action.get('speak_text') or '',
                'scene_summary': action.get('scene_summary') or '',
                'user_text': action.get('user_text') or '',
            })
            return bool(outcome.get('ok'))
        return False

    def find_target_from_text(self, text):
        normalized_text = to_simplified(text)
        for eng, aliases in self.objects.items():
            if eng.lower() in normalized_text.lower():
                return eng
            for alias in aliases:
                if to_simplified(alias) in normalized_text:
                    return eng
        return None

    def all_target_names(self):
        with self._lock:
            return sorted(self.objects.keys())

    def display_target_name(self, target):
        key = str(target or '').strip()
        if not key:
            return ''
        aliases = self.objects.get(key) or []
        for alias in aliases:
            normalized = to_simplified(alias)
            if any('\u4e00' <= ch <= '\u9fff' for ch in normalized):
                return normalized
        return key

    def resolve_target_object(self, text='', agent_result=None, allow_llm=True):
        result = agent_result or {}
        target = result.get('target_object')
        if target:
            return target

        user_text = text or result.get('user_text', '')
        direct = self.find_target_from_text(user_text)
        if direct:
            return direct

        context_target = (self.task_context or {}).get('target_object')
        if context_target and any(word in to_simplified(user_text) for word in ('它', '这个', '那个', '继续', '还是')):
            return context_target

        if not allow_llm or not self.rkllama or not self.rkllama.ensure_available():
            return None

        candidates = ', '.join(self.all_target_names())
        context_hint = context_target or '无'
        prompt = (
            '请从用户的话里提取要操作的物体，并映射成候选 YOLO 类名之一。'
            f'候选类名只有这些：{candidates}。'
            f'当前连续任务目标提示：{context_hint}。'
            '如果能确定，必须只输出一个英文类名；如果不能确定或不在候选中，只输出 NONE。'
            f'用户原话：{user_text}'
        )
        system_prompt = '你是一个只做目标类别映射的助手。输出必须是单个英文类名或 NONE，不要解释。'
        try:
            reply, _ = self.run_with_rkllama(
                '目标映射',
                lambda: self.rkllama.chat(prompt, system_prompt),
                pause_accelerators=True,
            )
        except Exception as exc:
            self.add_log(f'❌ LLM 目标映射失败: {exc}')
            return None

        normalized = reply.strip().splitlines()[0].strip().strip('`').strip().lower()
        self.add_log(f'🧠 LLM 目标映射结果: {normalized}')
        if normalized == 'none':
            return None
        for name in self.all_target_names():
            if normalized == name.lower():
                return name
        return None

    def build_agent_runtime_context(self):
        current_mode = self.current_mode()
        with self._lock:
            vector_payload = dict(self.vector_payload or {})
            obstacle_payload = dict(self.obstacle_payload or {})
            pending_action = dict(self.pending_agent_action or {})
            detection = dict(self.detection or {})
        vector_payload.setdefault('running', self.rk3588.running if self.rk3588 else False)
        vector_payload.setdefault('ready', self.rk3588.ready if self.rk3588 else False)
        obstacle_payload.setdefault('running', self.rk3588_obstacle.running if self.rk3588_obstacle else False)
        obstacle_payload.setdefault('ready', self.rk3588_obstacle.ready if self.rk3588_obstacle else False)
        return {
            'mode_name': current_mode.name if current_mode else '',
            'mode_index': self.mode_index,
            'current_target': self.target,
            'camera_available': bool(self.camera and self.camera.available),
            'camera_source': self.camera.active_source() if self.camera else '',
            'visual_source': self._agent_visual_source,
            'visual_summary': self._agent_visual_summary,
            'visual_attempted': bool(self._agent_visual_attempted),
            'visual_summary_source': 'intern' if self._agent_visual_summary else '',
            'yolo_available': bool(self.yolo and self.yolo.available),
            'stt_available': bool(self.stt and self.stt.available),
            'tts_available': bool(self.tts and self.tts.available),
            'vector_state': vector_payload,
            'obstacle_state': obstacle_payload,
            'last_detection': detection,
            'pending_action': pending_action,
            'task_context': dict(self.task_context or {}),
        }

    def capture_agent_snapshot(self, crop_left=False):
        frame = self.camera.get_frame_raw() if self.camera else None
        out_dir = Path(resolve_project_path('runtime_outputs/agent'))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / 'latest_scene.jpg'
        if frame is None and self.camera:
            try:
                self.camera.open_once()
            except Exception as exc:
                self.add_log(f'⚠️ 语音/Agent 单帧抓图失败: {exc}')
            frame = self.camera.get_frame_raw()
        if frame is not None:
            try:
                if crop_left and frame.shape[1] > frame.shape[0] * 2:
                    mid = frame.shape[1] // 2
                    frame = frame[:, :mid]
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                self._agent_visual_source = str(path)
                self.add_log(f'📸 抓取实时相机画面: {path}')
                return str(path)
            except Exception as exc:
                self.add_log(f'⚠️ 保存 agent 画面失败: {exc}')

        preview_candidates = [
            str(path),
            self.rk3588.preview_output_path if self.rk3588 else '',
            self.rk3588_obstacle.preview_output_path if self.rk3588_obstacle else '',
        ]
        for candidate in preview_candidates:
            if candidate and os.path.exists(candidate) and candidate.lower().endswith(('.jpg', '.jpeg', '.png')):
                self._agent_visual_source = candidate
                self.add_log(f'📸 抓取回退预览图: {candidate}')
                return candidate
        self._agent_visual_source = ''
        return None

    def pick_multimodal_warmup_image(self):
        candidates = [
            self.capture_agent_snapshot(crop_left=True),
            self.capture_agent_snapshot(crop_left=False),
            self.rk3588.preview_output_path if self.rk3588 else '',
            self.rk3588_obstacle.preview_output_path if self.rk3588_obstacle else '',
            resolve_project_path('runtime_outputs/agent/latest_scene.jpg'),
            resolve_project_path('runtime_outputs/rk3588/vector_preview.jpg'),
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def warmup_multimodal_pipeline(self):
        self.add_log('⏭️ 本地多模态视觉预热已禁用')
        return False
    def capture_agent_detections(self):
        if not (self.yolo and self.yolo.available and self.camera and self.camera.available):
            return []
        try:
            return self.yolo.detect(None)
        except Exception as exc:
            self.add_log(f'⚠️ Agent 检测失败: {exc}')
            return []

    def transcribe_audio(self, audio_path, log_prefix='语音'):
        text = ''
        try:
            if self.rkllama and self.rkllama.ensure_available():
                text, _ = self.run_with_rkllama(
                    f'{log_prefix}转写',
                    lambda: self.rkllama.transcribe(audio_path),
                    pause_accelerators=False,
                )
            else:
                text = self.stt.transcribe(audio_path)
        except Exception as exc:
            self.add_log(f'❌ {log_prefix}转写失败: {exc}')
            return ''
        normalized = to_simplified(text)
        if normalized:
            self.add_log(f'🗣️ {log_prefix}识别结果: {normalized}')
        else:
            self.add_log(f'⚠️ {log_prefix}转写返回空白')
        return normalized

    def analyze_voice_assistant_image(self, user_text, image_path):
        image_path = str(image_path or '').strip()
        if not image_path:
            self.add_log('⚠️ 语音助手未拿到当前画面')
            return '没有获取到画面'

        backend = str(self.config.get('voice_assistant', {}).get('vision_backend', 'api')).strip().lower()
        reply = ''
        if backend == 'api' and self.cloud_vision and self.cloud_vision.available():
            try:
                reply = self.cloud_vision.analyze_image(image_path, prompt=user_text)
                if reply:
                    self.add_log(f'🖼️ 云端视觉回答: {reply}')
                    return reply
            except Exception as exc:
                self.add_log(f'⚠️ 云端视觉分析失败: {exc}')
            return '云端视觉暂时不可用'

        self.add_log('⚠️ 语音助手云端视觉未配置，不使用本地视觉模型回退')
        return '云端视觉暂时不可用'

    def run_local_agent(self, user_text):
        image_path = self.capture_agent_snapshot()
        detections = self.capture_agent_detections()
        def _run():
            visual_summary = ''
            self._agent_visual_attempted = bool(image_path and self.cloud_vision and self.cloud_vision.available())
            if self._agent_visual_attempted:
                visual_summary = self.cloud_vision.analyze_image(image_path, prompt='请简短描述当前画面里最重要的内容。')
                self._agent_visual_summary = visual_summary
                if visual_summary:
                    self.add_log(f'🖼️ 云端画面摘要: {visual_summary}')
            else:
                self._agent_visual_summary = ''
                self._agent_visual_attempted = False
                if image_path:
                    self.add_log('⚠️ Agent 跳过视觉摘要：云端视觉不可用，不使用本地视觉模型')
            mode = self.current_mode()
            mode_name = mode.name if mode else ''
            runtime_context = self.build_agent_runtime_context()
            if not image_path:
                self.add_log('⚠️ Agent 未拿到当前相机图，尝试仅用语音与已有上下文推理')
            result = self.agent.run(
                user_text=user_text,
                mode_name=mode_name,
                image_path=image_path,
                detections=detections,
                candidate_targets=self.all_target_names(),
                runtime_context=runtime_context,
            )
            if visual_summary and not result.get('scene_summary'):
                result['scene_summary'] = visual_summary
            return result

        result = self.run_with_rkllama('Agent 推理', _run, pause_accelerators=True)
        self.set_agent_result(result)
        summary = result.get('scene_summary', '')
        intent = result.get('intent', 'unknown')
        target = result.get('target_object')
        self.add_log(f'🧠 Agent 意图: {intent} target={target or "-"}')
        if summary:
            self.add_log(f'🖼️ Agent 视觉总结: {summary}')
        return result

    def run_with_rkllama(self, task_name, func, pause_accelerators=False):
        with self._rkllama_call_lock:
            resume_state = None
            if pause_accelerators:
                resume_state = self._pause_accelerators_for_rkllama(task_name)
            try:
                return func()
            finally:
                if resume_state is not None:
                    self._resume_accelerators_after_rkllama(resume_state, task_name)

    def _pause_accelerators_for_rkllama(self, task_name):
        self._log_npu_policy_once()
        with self._lock:
            self._npu_exclusive_task = task_name
        resume_state = {
            'vector_running': bool(self.rk3588 and self.rk3588.running),
            'obstacle_running': bool(self.rk3588_obstacle and self.rk3588_obstacle.running),
            'target': self.target,
        }
        if resume_state['vector_running']:
            self.add_log(f'⏸️ {task_name}期间暂停 RK3588 辅助抓取进程，避免 NPU 冲突')
            self.rk3588.stop()
        if resume_state['obstacle_running']:
            self.add_log(f'⏸️ {task_name}期间暂停 RK3588 避障进程，避免 NPU 冲突')
            self.rk3588_obstacle.stop()
        self._wait_accelerators_stopped(task_name)
        if self._npu_pause_drain_s > 0:
            self.add_log(f'🧹 {task_name} 前等待 NPU 排空 {self._npu_pause_drain_s:.1f}s')
            time.sleep(self._npu_pause_drain_s)
        gc.collect()
        return resume_state

    def _resume_accelerators_after_rkllama(self, resume_state, task_name):
        try:
            current_mode = self.current_mode()
            mode_name = current_mode.name if current_mode else ''
            if self._npu_resume_delay_s > 0:
                time.sleep(self._npu_resume_delay_s)

            if resume_state.get('vector_running') and mode_name == '辅助抓取模式' and self.rk3588:
                self.rk3588.start()
                target = str(resume_state.get('target') or self.target or '').strip()
                if target:
                    self.rk3588.set_target_class(target)
                self.add_log(f'▶️ {task_name}完成，已恢复 RK3588 辅助抓取进程')

            if resume_state.get('obstacle_running') and mode_name == '导航识别模式' and self.rk3588_obstacle:
                self.rk3588_obstacle.start()
                self.add_log(f'▶️ {task_name}完成，已恢复 RK3588 避障进程')
        finally:
            with self._lock:
                if self._npu_exclusive_task == task_name:
                    self._npu_exclusive_task = ''

    def npu_exclusive_active(self):
        with self._lock:
            return bool(self._npu_exclusive_task)

    def mode_transitioning(self):
        with self._lock:
            return bool(self._mode_transitioning)

    def activate_vector_runtime(self, reason=''):
        if self.rk3588_obstacle and self.rk3588_obstacle.running:
            stopped = self.rk3588_obstacle.stop()
            self.add_log('🛑 已关闭 RK3588 避障进程' if stopped else '⚠️ RK3588 避障进程未能完全关闭')
            if not stopped:
                return False
        if self.rk3588 and not self.rk3588.running:
            self.rk3588.start()
        return bool(self.rk3588 and self.rk3588.running)

    def activate_obstacle_runtime(self, reason=''):
        if self.rk3588 and self.rk3588.running:
            stopped = self.rk3588.stop()
            self.add_log('🛑 已关闭辅助抓取 RK3588 向量进程' if stopped else '⚠️ 辅助抓取 RK3588 向量进程未能完全关闭')
            if not stopped:
                return False
        if self.rk3588_obstacle and not self.rk3588_obstacle.running:
            self.rk3588_obstacle.start()
        return bool(self.rk3588_obstacle and self.rk3588_obstacle.running)

    def pause_realtime_runtimes(self, reason=''):
        ok = True
        if self.rk3588 and self.rk3588.running:
            stopped = self.rk3588.stop()
            self.add_log('🛑 已关闭辅助抓取 RK3588 向量进程' if stopped else '⚠️ 辅助抓取 RK3588 向量进程未能完全关闭')
            ok = ok and stopped
        if self.rk3588_obstacle and self.rk3588_obstacle.running:
            stopped = self.rk3588_obstacle.stop()
            self.add_log('🛑 已关闭避障进程' if stopped else '⚠️ 避障进程未能完全关闭')
            ok = ok and stopped
        return ok

    def _wait_accelerators_stopped(self, task_name):
        deadline = time.time() + max(0.1, self._npu_stop_timeout_s)
        while time.time() < deadline:
            vector_running = bool(self.rk3588 and self.rk3588.running)
            obstacle_running = bool(self.rk3588_obstacle and self.rk3588_obstacle.running)
            if not vector_running and not obstacle_running:
                return True
            time.sleep(0.1)
        vector_running = bool(self.rk3588 and self.rk3588.running)
        obstacle_running = bool(self.rk3588_obstacle and self.rk3588_obstacle.running)
        self.add_log(
            f'⚠️ {task_name} 前等待加速进程退出超时: vector_running={vector_running} obstacle_running={obstacle_running}'
        )
        return False

    def _log_npu_policy_once(self):
        if self._npu_policy_logged:
            return
        rkllama_core_mask = ((self.config.get('rkllama', {}) or {}).get('env', {}) or {}).get('RKLLAMA_RKNN_CORE_MASK', 'default')
        self.add_log(
            '🧩 NPU 调度策略: RK3588 实时链路保持多核运行，rkllama 使用串行调用；'
            f'LLM 前暂停实时链路并排空 {self._npu_pause_drain_s:.1f}s，'
            f'恢复前等待 {self._npu_resume_delay_s:.1f}s，rkllama_core_mask={rkllama_core_mask}'
        )
        self._npu_policy_logged = True

    def execute_assist_grab_target(self, target, agent_result=None):
        target = str(target or '').strip()
        if not target:
            return {'ok': False, 'target': '', 'message': '未提供目标物体'}

        result = agent_result or {}
        display_target = self.display_target_name(target) or target
        self.set_target(target)
        self.update_task_context({
            'intent': 'assist_grab',
            'mode_name': '辅助抓取模式',
            'target_object': target,
            'user_text': result.get('user_text', ''),
            'obstacle_label': '',
        })
        self.add_log(f'🎯 最终辅助抓取目标: {target}')

        if self.rk3588 and self.rk3588.set_target_class(target):
            msg = result.get('speak_text') or f'开始辅助抓取{display_target}'
            self.add_log(f'🧭 已发送辅助抓取目标到 RK3588: {target}')
            self.speak_text(msg)
            return {'ok': True, 'target': target, 'message': msg}

        msg = result.get('speak_text') or '目标已识别，但下发失败'
        self.add_log('❌ 写入 RK3588 目标失败')
        self.speak_text(msg)
        return {'ok': False, 'target': target, 'message': msg}

    def execute_obstacle_reply(self, agent_result=None):
        result = agent_result or {}
        obstacle = (self.obstacles or [{}])[0] if self.obstacles else {}
        speak_text = result.get('speak_text') or result.get('scene_summary')
        self.clear_object_focus_state()
        if not speak_text:
            speak_text = self._build_obstacle_speech(obstacle) or '当前未检测到明显障碍'
        self.update_task_context({
            'intent': result.get('intent') or 'obstacle_query',
            'mode_name': '导航识别模式',
            'user_text': result.get('user_text', ''),
            'obstacle_label': obstacle.get('label') or '',
            'target_object': '',
        })
        self.add_log(f'🧠 避障回复: {speak_text}')
        self.speak_text(speak_text)
        return {'ok': True, 'message': speak_text}

    def execute_general_reply(self, agent_result=None, default_intent='chat'):
        result = agent_result or {}
        current_mode = self.current_mode()
        intent = result.get('intent') or default_intent
        speak_text = result.get('speak_text') or result.get('scene_summary') or '我已收到'
        self.clear_object_focus_state()
        self.update_task_context({
            'intent': intent,
            'mode_name': result.get('mode_name') or (current_mode.name if current_mode else None),
            'user_text': result.get('user_text', ''),
            'target_object': '',
            'obstacle_label': '',
        })
        label = '场景理解' if intent == 'scene_explain' else '对话'
        self.add_log(f'🧠 {label}回复: {speak_text}')
        self.speak_text(speak_text)
        return {'ok': True, 'message': speak_text}

    def _format_distance_m(self, distance_m):
        try:
            value = float(distance_m)
        except Exception:
            return ''
        if value >= 10.0:
            text = str(round(value, 1))
        else:
            text = str(round(value, 2))
        return f'{self._number_text_to_chinese(text)}米'

    def _number_text_to_chinese(self, text):
        digit_map = {
            '0': '零',
            '1': '一',
            '2': '二',
            '3': '三',
            '4': '四',
            '5': '五',
            '6': '六',
            '7': '七',
            '8': '八',
            '9': '九',
            '.': '点',
            '-': '负',
        }
        return ''.join(digit_map.get(ch, ch) for ch in str(text))

    def _obstacle_direction_text(self, vector_x_m):
        try:
            x_m = float(vector_x_m)
        except Exception:
            return '前方'
        if x_m <= -0.18:
            return '左侧前方'
        if x_m <= -0.06:
            return '左前方'
        if x_m < 0.06:
            return '正前方'
        if x_m < 0.18:
            return '右前方'
        return '右侧前方'

    def _obstacle_label_text(self, label):
        text = str(label or '').strip()
        if not text:
            return '障碍物'
        return self._OBSTACLE_LABELS.get(text, text)

    def _build_obstacle_speech(self, obstacle):
        obstacle = obstacle or {}
        label = self._obstacle_label_text(obstacle.get('label') or obstacle.get('obstacle_class_name'))
        if not label:
            return ''
        direction = self._obstacle_direction_text(obstacle.get('vector_x_m'))
        distance_text = self._format_distance_m(obstacle.get('distance_m'))
        if distance_text:
            return f'{direction}有{label}，距离{distance_text}'
        return f'{direction}有{label}'

    def _should_speak_obstacle(self, obstacle, force=False):
        if force:
            return True
        distance_m = obstacle.get('distance_m')
        try:
            distance_value = float(distance_m)
        except Exception:
            return False
        now = time.time()
        with self._speech_async_lock:
            speech_busy = self._speech_async_busy
            speech_last_done_ts = self._speech_async_last_done_ts
        if speech_busy:
            return False
        last_speech_boundary = max(self._last_obstacle_speak_ts, speech_last_done_ts)
        if distance_value < 3.0 and now - last_speech_boundary >= 5.0:
            self._last_obstacle_speech_key = (
                f'{obstacle.get("label")}:'
                f'{self._obstacle_direction_text(obstacle.get("vector_x_m"))}:'
                f'{round(distance_value, 1)}'
            )
            self._last_obstacle_speak_ts = now
            return True
        return False

    def apply_obstacle_payload(self, payload, speak=False):
        payload = payload or {}
        self.set_obstacle_payload(payload)
        obstacle_name = payload.get('obstacle_class_name')
        distance_m = payload.get('distance_m')
        if obstacle_name:
            obs = [{
                'label': obstacle_name,
                'distance_m': distance_m,
                'vector_x_m': payload.get('vector_x_m'),
                'vector_z_m': payload.get('vector_z_m'),
            }]
            self.set_obstacles(obs)
            self.update_task_context({
                'intent': 'obstacle_monitor',
                'mode_name': '导航识别模式',
                'obstacle_label': obstacle_name,
            })
            should_speak = self._should_speak_obstacle(obs[0], force=speak)
            if should_speak:
                msg = self._build_obstacle_speech(obs[0])
                self.add_log(f'🔊 避障播报: {msg}')
                self.speak_text_async(msg, reason='避障播报')
            return {'ok': True, 'obstacle': obs[0]}
        self.set_obstacles([])
        self.update_task_context({
            'obstacle_label': '',
        })
        return {'ok': True, 'obstacle': None}

    def agent_status_snapshot(self):
        with self._lock:
            pending = dict(self.pending_agent_action or {})
            task_ctx = dict(self.task_context or {})
            visual_source = self._agent_visual_source
            visual_summary = self._agent_visual_summary
        return {
            'visual_source': visual_source,
            'visual_summary': visual_summary,
            'pending_action': pending,
            'pending_intent': pending.get('intent'),
            'pending_mode_name': pending.get('mode_name'),
            'pending_target_object': pending.get('target_object'),
            'task_context': task_ctx,
            'mode_names': self.mode_names(),
            'runtime_context': self.build_agent_runtime_context(),
        }

    def execute_agent_result(self, result):
        result = result or {}
        intent = result.get('intent')
        current_mode_name = self.current_mode().name if self.current_mode() else None
        if not result.get('mode_name'):
            result['mode_name'] = self._infer_mode_name_from_intent(intent)
        if result.get('mode_name') and result.get('mode_name') != current_mode_name:
            result['should_switch_mode'] = True
        else:
            result['should_switch_mode'] = False
        if not result.get('target_object') and intent == 'assist_grab':
            resolved_target = self.resolve_target_object(
                text=result.get('user_text', ''),
                agent_result=result,
                allow_llm=True,
            )
            if resolved_target:
                result['target_object'] = resolved_target
        if not result.get('target_object'):
            fallback_target = self.find_target_from_text(result.get('user_text', ''))
            if fallback_target:
                result['target_object'] = fallback_target

        action = {
            'intent': intent,
            'mode_name': result.get('mode_name'),
            'target_object': result.get('target_object'),
            'speak_text': result.get('speak_text'),
            'scene_summary': result.get('scene_summary'),
            'user_text': result.get('user_text'),
            'confidence': result.get('confidence'),
        }
        target = result.get('target_object')
        if target and intent == 'assist_grab':
            self.set_target(target)
        elif intent in {'obstacle_query', 'scene_explain', 'chat'}:
            self.clear_object_focus_state()

        if intent in {'assist_grab', 'obstacle_query', 'scene_explain', 'chat'}:
            task_update = {
                'intent': intent,
                'mode_name': result.get('mode_name'),
                'user_text': result.get('user_text'),
            }
            if target:
                task_update['target_object'] = target
            elif intent in {'scene_explain', 'chat'}:
                task_update['target_object'] = ''
            if intent != 'obstacle_query':
                task_update['obstacle_label'] = ''
            self.update_task_context(task_update)

        switched = False
        if result.get('should_switch_mode') and result.get('mode_name'):
            self.set_pending_agent_action(action)
            if self.set_mode_by_name(result.get('mode_name')):
                switched = True
                self.add_log(f'🔀 Agent 自动切换到: {result.get("mode_name")}')
            else:
                self.add_log(f'⚠️ Agent 请求切换模式失败: {result.get("mode_name")}')
                self.set_pending_agent_action(None)

        if not switched and intent in {'assist_grab', 'obstacle_query', 'scene_explain', 'chat'}:
            self.set_pending_agent_action(action)
            self.consume_pending_agent_action()
        return result

    def should_execute_agent_in_current_mode(self, result, allowed_intents):
        result = result or {}
        allowed = set(allowed_intents or [])
        intent = result.get('intent') or 'unknown'
        if result.get('should_switch_mode'):
            return False
        if intent in allowed:
            return True
        return intent == 'unknown'

    def route_agent_result_for_mode(self, result, allowed_intents):
        if self.should_execute_agent_in_current_mode(result, allowed_intents):
            return True
        self.execute_agent_result(result)
        return False

    def _infer_mode_name_from_intent(self, intent):
        if intent == 'assist_grab':
            return '辅助抓取模式'
        if intent == 'obstacle_query':
            return '导航识别模式'
        if intent in {'scene_explain', 'chat'}:
            mode = self.current_mode()
            return mode.name if mode else None
        return None

    def _process_agent_input_text(self, text, empty_log, execute_result=True, allow_scene_explain=True):
        normalized_text = to_simplified(text)
        self.set_stt_text(normalized_text)
        if not normalized_text:
            self.add_log(empty_log)
            return {'ok': False, 'text': '', 'agent_result': {}, 'target': None, 'scene_explain': False}

        self.add_log(f'🗣️ 识别文本: {normalized_text}')
        agent_result = self.run_local_agent(normalized_text)
        if execute_result:
            self.execute_agent_result(agent_result)
        target = agent_result.get('target_object')
        if allow_scene_explain and agent_result.get('intent') == 'scene_explain':
            return {
                'ok': True,
                'text': normalized_text,
                'agent_result': agent_result,
                'target': None,
                'scene_explain': True,
            }
        return {
            'ok': True,
            'text': normalized_text,
            'agent_result': agent_result,
            'target': target,
            'scene_explain': False,
        }

    def process_agent_command(self, audio_path, log_prefix='语音', allow_scene_explain=True, execute_result=True):
        text = self.transcribe_audio(audio_path, log_prefix=log_prefix)
        return self._process_agent_input_text(
            text,
            empty_log=f'❓ 未识别到{log_prefix}内容',
            execute_result=execute_result,
            allow_scene_explain=allow_scene_explain,
        )

    def process_agent_text(self, text, execute_result=True, allow_scene_explain=True):
        return self._process_agent_input_text(
            text,
            empty_log='❓ 未识别到有效文本内容',
            execute_result=execute_result,
            allow_scene_explain=allow_scene_explain,
        )

    def process_agent_audio_release(self, audio_path, log_prefix, too_short_message):
        if not audio_path:
            self.add_log(too_short_message)
            return {'ok': False, 'text': '', 'agent_result': {}, 'target': None, 'scene_explain': False}
        return self.process_agent_command(audio_path, log_prefix=log_prefix, execute_result=False)

    def handle_mode_agent_release(
        self,
        audio_path,
        *,
        log_prefix,
        too_short_message,
        allowed_intents,
        mode_handler,
        empty_speak_text='',
        unsupported_target_text='',
        unsupported_target_log='',
        allow_target_llm=False,
    ):
        processed = self.process_agent_audio_release(
            audio_path,
            log_prefix=log_prefix,
            too_short_message=too_short_message,
        )
        if not processed.get('ok'):
            if empty_speak_text and not processed.get('text'):
                self.speak_text(empty_speak_text)
            return {'ok': False, 'processed': processed}

        text = processed.get('text', '')
        agent_result = processed.get('agent_result', {})
        if not self.route_agent_result_for_mode(agent_result, allowed_intents):
            return {'ok': True, 'processed': processed, 'routed_elsewhere': True}

        outcome = mode_handler(processed, text, agent_result, allow_target_llm, unsupported_target_text, unsupported_target_log)
        return {'ok': bool(outcome.get('ok')), 'processed': processed, 'outcome': outcome}

    def handle_assist_grab_release(self, processed, text, agent_result, allow_target_llm, unsupported_target_text, unsupported_target_log):
        if agent_result.get('intent') in {'scene_explain', 'chat'}:
            return self.execute_general_reply(agent_result)
        target = self.resolve_target_object(text=text, agent_result=agent_result, allow_llm=allow_target_llm)
        if not target:
            log_message = unsupported_target_log or '❌ 未从语音中提取出 YOLO 目标物体'
            if text:
                log_message = f'{log_message}: {text}'
            self.add_log(log_message)
            self.speak_text(agent_result.get('speak_text') or unsupported_target_text or '未提取出目标物体')
            return {'ok': False, 'message': 'no_target'}
        return self.execute_assist_grab_target(target, {
            **agent_result,
            'user_text': text,
        })

    def handle_obstacle_release(self, processed, text, agent_result, allow_target_llm, unsupported_target_text, unsupported_target_log):
        if agent_result.get('intent') in {'chat', 'scene_explain'}:
            return self.execute_general_reply(agent_result, default_intent='chat')
        return self.execute_obstacle_reply(agent_result)

    def abort_active_recording(self):
        with self._lock:
            if not self._s2_active:
                return False
        try:
            self.recorder.stop()
        except Exception as exc:
            self.add_log(f'⚠️ 停止当前录音失败: {exc}')
        with self._lock:
            self._s2_active = False
            self._s2_pressed_mode_index = None
        return True

    def reset_runtime_state(self, speak=True):
        self._resetting_runtime = True
        self.abort_active_recording()
        with self._lock:
            self._s2_active = False
            self._s2_pressed_mode_index = None
            self.stt_text = ''
        self.clear_object_focus_state()
        self.set_obstacles([])
        self.set_obstacle_payload({})
        self.set_vector_payload({})
        self.clear_task_context()
        self.set_pending_agent_action(None)
        self.set_agent_result({})
        self._agent_visual_source = ''
        self._agent_visual_summary = ''
        self._agent_visual_attempted = False
        try:
            self.set_mode(0)
        finally:
            self._resetting_runtime = False
        self.add_log('🔄 已重置')
        if speak:
            self.speak_text('已重置')

    def _cleanup_old_files(self):
        base = Path(resolve_project_path('runtime_outputs'))
        max_age = 300
        now = time.time()
        for root, dirs, files in os.walk(base):
            for name in files:
                path = os.path.join(root, name)
                try:
                    if now - os.path.getmtime(path) > max_age:
                        os.remove(path)
                except Exception:
                    pass

    def _cleanup_loop(self):
        while self._cleanup_running:
            time.sleep(60)
            try:
                self._cleanup_old_files()
            except Exception:
                pass
            gc.collect()

    def on_rk3588_ready(self):
        self.add_log('🔔 本地模型加载完成')

    def on_glasses_button_press(self):
        self.waiting_for_glasses_button = False
        self.handle_s2_press()

    def on_glasses_button_release(self):
        self.handle_s2_release()

    def handle_s2_press(self):
        with self._lock:
            if self._s2_active:
                self.add_log('⏭️ 忽略重复的录音按下事件')
                return False
            mode_index = self.mode_index
        mode = self.modes[mode_index] if self.modes else None
        if mode:
            started = bool(mode.on_s2_press())
            if not started:
                self.add_log('⚠️ 录音未成功开始，忽略本次按下')
                return False
            with self._lock:
                self._s2_active = True
                self._s2_pressed_mode_index = mode_index
                self._last_talk_event_ts = time.time()
            return True
        return False

    def handle_s2_release(self):
        with self._lock:
            if not self._s2_active:
                self.add_log('⏭️ 忽略未开始录音的松开事件')
                return False
            target_index = self._s2_pressed_mode_index if self._s2_pressed_mode_index is not None else self.mode_index
            self._s2_active = False
            self._s2_pressed_mode_index = None
            self._last_talk_event_ts = time.time()
        if not self.modes:
            return False
        if target_index < 0 or target_index >= len(self.modes):
            return False
        self.modes[target_index].on_s2_release()
        return True

    def _handle_glasses_event(self, kind, data):
        if kind == 'disconnect':
            self.waiting_for_glasses_button = False
            with self._lock:
                self._s2_active = False
                self._s2_pressed_mode_index = None
            return
        if kind in {'hello', 'status'}:
            self.waiting_for_glasses_button = True
            return
        if kind in {'mode_short', 'button_mode_short', 'mode_down', 'button_mode_down'}:
            if not self._accept_glasses_buttons:
                self.add_log('⏭️ 忽略启动阶段的 MODE 事件')
                return
            now = time.time()
            with self._lock:
                if self._s2_active:
                    self.add_log('⏭️ 录音进行中，忽略 MODE 事件')
                    return
                if now - self._last_mode_event_ts < self._mode_event_min_interval_s:
                    self.add_log('⏭️ 忽略抖动的 MODE 事件')
                    return
                if now - self._last_talk_event_ts < self._ignore_mode_after_talk_s:
                    self.add_log('⏭️ TALK 事件后短时间内忽略 MODE 事件')
                    return
                self._last_mode_event_ts = now
            self.add_log('🕹️ 收到 MODE 切换事件')
            self.cycle_mode()
            return
        if kind == 'button_down':
            if not self._accept_glasses_buttons:
                self.add_log('⏭️ 忽略启动阶段的 TALK 按下事件')
                return
            self.waiting_for_glasses_button = False
            self.add_log('🕹️ 收到 TALK 按下事件')
            self.on_glasses_button_press()
        elif kind == 'button_up':
            if not self._accept_glasses_buttons:
                self.add_log('⏭️ 忽略启动阶段的 TALK 松开事件')
                return
            self.add_log('🕹️ 收到 TALK 松开事件')
            self.on_glasses_button_release()


def _init_audio():
    subprocess.run(['amixer', '-c', '4', 'sset', 'Headphone', '3', 'unmute'], stderr=subprocess.DEVNULL)
    subprocess.run(['amixer', '-c', '4', 'sset', 'Headphone Mixer', '11'], stderr=subprocess.DEVNULL)
    subprocess.run(['amixer', '-c', '4', 'sset', 'DAC', '192'], stderr=subprocess.DEVNULL)
    subprocess.run(['amixer', '-c', '4', 'sset', 'Speaker', 'on'], stderr=subprocess.DEVNULL)


def create_app():
    _init_audio()
    state = AppState()
    state.modes = [
        AssistGrabMode(state),
        NavRecognitionMode(state),
        VoiceAssistantMode(state),
    ]

    state.add_log('🚀 应用启动中...')
    state.glasses.start()
    state.wristband.start()
    state.waiting_for_glasses_button = bool(state.glasses and state.glasses.connected)

    camera_managed_by_rk3588 = bool(state.rk3588 and state.rk3588.enabled)
    if camera_managed_by_rk3588:
        state.camera.open_once()
        state.add_log('📷 摄像头由 RK3588 向量进程使用，主程序仅抓取初始预览帧')
    else:
        state.camera.start()
        if state.camera.check():
            source = state.camera.active_source()
            state.add_log(f'📷 摄像头已初始化: {source}')
        else:
            retry_interval = state.config.get('camera', {}).get('retry_interval_s', 3)
            state.add_log(f'⚠️ 开机未检测到摄像头，正在每 {retry_interval} 秒重试')

    state.stt.check()
    state.add_log('🎙️ STT 初始化完成' if state.stt._available else '⚠️ STT 未安装 (pip install openai-whisper)')

    state.tts.check()
    state.add_log('🔊 TTS 初始化完成' if state.tts._available else '⚠️ TTS 未安装 (apt install espeak)')

    state.add_log('⏭️ 本地 Intern 视觉分析通路已禁用')
    state.add_log('🧠 正在检查本地 rkllama 服务...')
    state.add_log('🧠 本地 rkllama/internvl 服务' + ('已就绪' if state.rkllama.ensure_available() else '未连接'))
    cloud_vision_ready = bool(state.cloud_vision and state.cloud_vision.available())
    backend_name = str(state.config.get('voice_assistant', {}).get('vision_backend', 'api')).strip().lower() or 'api'
    state.add_log(
        f'🖼️ 语音助手视觉后端: {backend_name}'
        + ('（云端可用）' if cloud_vision_ready else '（云端不可用）')
    )
    state.yolo.check()
    if state.yolo._available:
        state.yolo.start_background(state.camera)
    state.add_log('👁️ YOLO' + ('已就绪（后台连续检测）' if state.yolo._available else '未安装 (pip install ultralytics)'))
    if state.glasses.enabled:
        if state.glasses.connected:
            state.add_log('🎧 Glasses SDK 录音/按键已接管')
        else:
            state.add_log('⚠️ Glasses SDK 未连接，主程序录音将无法响应按键2')

    state.add_log('🔍 设备自检中...')
    set_camera_state(state.camera)
    state.device_status = check_all_devices()
    for name, info in state.device_status['checks'].items():
        icon = '✅' if info['ok'] else '❌'
        state.add_log(f'{icon} {name}: {info["detail"]}')

    state.set_mode(0)
    if state.waiting_for_glasses_button:
        state.add_log('🕹️ 开机默认进入辅助抓取模式，等待眼镜 TALK 按钮')
    state._accept_glasses_buttons = True
    state.add_log('🕹️ 已开始接收眼镜 TALK 按键事件')

    def mode_update_loop():
        camera_was_available = state.camera.available
        while True:
            try:
                camera_available = state.camera.available
                if state._camera_started_by_mode:
                    if camera_available and not camera_was_available:
                        source = state.camera.active_source()
                        state.add_log(f'📷 摄像头已接入并开始工作: {source}')
                    elif not camera_available and camera_was_available:
                        retry_interval = state.config.get('camera', {}).get('retry_interval_s', 3)
                        state.add_log(f'⚠️ 摄像头断开，正在每 {retry_interval} 秒重试')
                camera_was_available = camera_available

                mode = state.current_mode()
                if mode and not state.mode_transitioning():
                    mode.update()
            except Exception:
                pass
            time.sleep(0.1)

    threading.Thread(target=mode_update_loop, daemon=True).start()

    state._cleanup_running = True
    state._cleanup_thread = threading.Thread(target=state._cleanup_loop, daemon=True)
    state._cleanup_thread.start()

    state.add_log('✅ 应用启动完成')
    state.add_log('⚡ 线程池(4核) + 后台YOLO检测 + 自动内存清理(60s周期) 已启用')
    return state
