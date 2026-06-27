import os
import subprocess
import base64
import time
from flask import Flask, jsonify, request, send_from_directory
from app.main import create_app

state = None
_sw_states = {'s1': False, 's2': False, 's3': False}
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
os.makedirs(static_dir, exist_ok=True)
flask_app = Flask(__name__, static_folder=static_dir)
_IMAGE_MAX_AGE_S = 2.5


def _pulse_switch(name, hold_s=0.2):
    _sw_states[name] = True

    def _reset():
        _sw_states[name] = False

    import threading
    threading.Timer(hold_s, _reset).start()


def _attach_runtime_hooks(app_state):
    if not app_state or not app_state.glasses:
        return

    def _mirror_switch_state(kind, data):
        if kind == 'disconnect':
            _sw_states['s2'] = False

    app_state.glasses.add_event_handler(_mirror_switch_state)


@flask_app.route('/')
def index():
    return send_from_directory(static_dir, 'index.html')


_audio_sink_cache = ''
_audio_sink_ts = 0

def _read_fresh_image_base64(path, max_age_s=_IMAGE_MAX_AGE_S):
    if not path or not os.path.exists(path):
        return None
    try:
        if time.time() - os.path.getmtime(path) > max_age_s:
            return None
        with open(path, 'rb') as handle:
            return base64.b64encode(handle.read()).decode('ascii')
    except Exception:
        return None


def _maybe_restart_stale_runtime(runtime, label, max_age_s=8.0):
    if not runtime:
        return
    try:
        age_s = runtime.output_age_s()
    except Exception:
        age_s = None
    if age_s is not None and age_s >= max_age_s:
        state.add_log(f'⚠️ {label}预览/向量输出停更 {age_s:.1f}s，等待模式 watchdog 处理')


def _camera_status_for_mode():
    if not state:
        return False, '', {}
    mode = state.current_mode()
    mode_name = mode.name if mode else ''
    runtime = None
    source = ''
    if mode_name == '辅助抓取模式':
        runtime = state.rk3588
        source = 'rk3588_vector'
    elif mode_name == '导航识别模式':
        runtime = state.rk3588_obstacle
        source = 'rk3588_obstacle'
    if runtime and runtime.running:
        age_s = runtime.output_age_s()
        if age_s is not None:
            return age_s <= _IMAGE_MAX_AGE_S, source, {
                'available': age_s <= _IMAGE_MAX_AGE_S,
                'active_source': source,
                'frame_age_ms': round(age_s * 1000.0, 1),
                'reconnect_attempts': 0,
            }
    diagnostics = state.camera.diagnostics() if state.camera else {}
    return bool(state.camera and state.camera.available), diagnostics.get('active_source') or '', diagnostics


@flask_app.route('/api/status')
def api_status():
    global _audio_sink_cache, _audio_sink_ts
    mode = state.current_mode()
    camera_available, camera_source, camera_diagnostics = _camera_status_for_mode()
    audio_sink = _audio_sink_cache
    if not audio_sink or time.time() - _audio_sink_ts > 5:
        try:
            out = subprocess.check_output(['pactl', 'info'], stderr=subprocess.DEVNULL, timeout=2)
            for line in out.decode().split('\n'):
                if 'Default Sink' in line:
                    audio_sink = line.split(':', 1)[-1].strip()
                    break
            _audio_sink_cache = audio_sink
            _audio_sink_ts = time.time()
        except Exception:
            pass
    return jsonify({
        'mode': mode.name if mode else 'unknown',
        'mode_index': state.mode_index,
        'mode_names': state.mode_names(),
        'voice_assistant': {
            'vision_backend': str(state.config.get('voice_assistant', {}).get('vision_backend', 'api')).strip().lower() or 'api',
            'cloud_vision_enabled': bool(state.cloud_vision.enabled) if state and state.cloud_vision else False,
            'cloud_vision_available': bool(state.cloud_vision.available()) if state and state.cloud_vision else False,
        },
        'device_status': state.device_status,
        'stt_text': state.stt_text,
        'target': state.target,
        'detection': state.detection,
        'obstacles': state.obstacles[-5:] if state.obstacles else [],
        'obstacle_payload': state.obstacle_payload,
        'logs': state.logs[-100:],
        'switches': {
            **dict(_sw_states),
            's2': bool(state.s2_active) if state else False,
        },
        'camera_available': camera_available,
        'camera_source': camera_source,
        'camera_diagnostics': camera_diagnostics,
        'yolo_available': state.yolo.available if state.yolo else False,
        'stt_available': state.stt.available if state.stt else False,
        'tts_available': state.tts.available if state.tts else False,
        'audio_sink': audio_sink,
        'vector_payload': state.vector_payload,
        'agent_result': state.agent_result,
        'agent_status': state.agent_status_snapshot(),
        'rk3588': {
            'enabled': state.rk3588.enabled if state.rk3588 else False,
            'running': state.rk3588.running if state.rk3588 else False,
            'ready': state.rk3588.ready if state.rk3588 else False,
            'target_input_path': state.rk3588.target_input_path if state.rk3588 else '',
            'preview_output_path': state.rk3588.preview_output_path if state.rk3588 else '',
            'rolling_output_path': state.rk3588.rolling_output_path if state.rk3588 else '',
            'history_output_path': state.rk3588.history_output_path if state.rk3588 else '',
            'csv_output_path': state.rk3588.csv_output_path if state.rk3588 else '',
            'last_error': state.rk3588.last_error if state.rk3588 else '',
        },
        'rk3588_obstacle': {
            'enabled': state.rk3588_obstacle.enabled if state.rk3588_obstacle else False,
            'running': state.rk3588_obstacle.running if state.rk3588_obstacle else False,
            'ready': state.rk3588_obstacle.ready if state.rk3588_obstacle else False,
            'rolling_output_path': state.rk3588_obstacle.rolling_output_path if state.rk3588_obstacle else '',
            'history_output_path': state.rk3588_obstacle.history_output_path if state.rk3588_obstacle else '',
            'csv_output_path': state.rk3588_obstacle.csv_output_path if state.rk3588_obstacle else '',
            'preview_output_path': state.rk3588_obstacle.preview_output_path if state.rk3588_obstacle else '',
            'last_error': state.rk3588_obstacle.last_error if state.rk3588_obstacle else '',
        },
        'glasses': {
            'enabled': state.glasses.enabled if state.glasses else False,
            'available': state.glasses.available if state.glasses else False,
            'connected': state.glasses.connected if state.glasses else False,
            'port': state.glasses.port if state.glasses else '',
            'waiting_for_press': state.waiting_for_glasses_button,
            'last_recording': state.glasses.last_wav_path if state.glasses else '',
            'last_error': state.glasses.last_error if state.glasses else '',
        },
        'wristband': state.wristband.status_snapshot() if state and state.wristband else {},
    })


@flask_app.route('/api/s1', methods=['POST'])
def api_s1():
    _pulse_switch('s1')
    state.cycle_mode()
    return jsonify({'ok': True})


@flask_app.route('/api/set_mode', methods=['POST'])
def api_set_mode():
    data = request.get_json(silent=True) or {}
    mode_name = str((data or {}).get('mode', '')).strip()
    mode_index = data.get('mode_index')
    ok = False
    if mode_name:
        ok = bool(state.set_mode_by_name(mode_name))
    elif mode_index is not None:
        try:
            state.set_mode(int(mode_index))
            ok = True
        except Exception:
            ok = False
    return jsonify({
        'ok': ok,
        'mode': state.current_mode().name if state.current_mode() else '',
        'mode_index': state.mode_index,
    })


@flask_app.route('/api/s2', methods=['POST'])
def api_s2():
    data = request.get_json(silent=True) or {}
    action = data.get('action', 'press')
    if action == 'press':
        _sw_states['s2'] = True
        if not state.handle_s2_press():
            _sw_states['s2'] = False
    elif action == 'release':
        _sw_states['s2'] = False
        state.handle_s2_release()
    return jsonify({'ok': True})


@flask_app.route('/api/tts_speak', methods=['POST'])
def api_tts_speak():
    data = request.get_json(silent=True) or {}
    text = (data or {}).get('text', '').strip()
    if text:
        state.add_log(f'🔊 TTS: {text}')
        return jsonify({'ok': bool(state.speak_text(text))})
    return jsonify({'ok': False})


@flask_app.route('/api/agent_test', methods=['POST'])
def api_agent_test():
    data = request.get_json() or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'ok': False, 'error': 'empty_text'})
    processed = state.process_agent_text(text, execute_result=True, allow_scene_explain=True)
    return jsonify({'ok': bool(processed.get('ok')), 'result': processed.get('agent_result', {}), 'processed': processed})


@flask_app.route('/api/voice_assistant_test', methods=['POST'])
def api_voice_assistant_test():
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip() or '我面前有什么'
    crop_left = bool(data.get('crop_left', True))
    speak = bool(data.get('speak', False))
    image_path = state.capture_agent_snapshot(crop_left=crop_left)
    reply = state.analyze_voice_assistant_image(text, image_path)
    if speak and reply:
        state.speak_text(reply)
    return jsonify({
        'ok': bool(reply),
        'text': text,
        'image_path': image_path,
        'reply': reply,
    })


@flask_app.route('/api/wristband_test', methods=['POST'])
def api_wristband_test():
    data = request.get_json(silent=True) or {}
    direction = str(data.get('direction', 'right')).strip().lower()
    mode = str(data.get('mode', 'grasp')).strip().lower()
    if mode not in {'grasp', 'avoid'}:
        mode = 'grasp'
    vector_x = 0.0
    vector_y = 0.0
    vector_z = 0.0
    status = 'tracking'
    active = True
    distance = 0.12
    if direction == 'left':
        vector_x = -0.12
    elif direction == 'up':
        vector_y = 0.12
    elif direction == 'down':
        vector_y = -0.12
    elif direction == 'front':
        vector_z = 0.12
    elif direction == 'back':
        vector_z = -0.12
    elif direction == 'stop':
        status = 'stop'
        active = False
        distance = 0.03
    else:
        vector_x = 0.12
    packet = {
        'timestamp_s': round(time.time(), 6),
        'frame_index': int(data.get('frame_index', 1) or 1),
        'status': status,
        'vector_active': active,
        'mode': mode,
        'vector_x_m': vector_x,
        'vector_y_m': vector_y,
        'vector_z_m': vector_z,
        'distance_m': distance,
    }
    if state.wristband:
        state.wristband.update_packet(packet)
    return jsonify({'ok': bool(state.wristband), 'packet': packet})


@flask_app.route('/api/set_volume', methods=['POST'])
def api_set_volume():
    data = request.get_json(silent=True) or {}
    vol = (data or {}).get('volume', 100)
    vol = max(0, min(100, vol))
    subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', f'{vol}%'], stderr=subprocess.DEVNULL)
    subprocess.run(['pactl', 'set-sink-mute', '@DEFAULT_SINK@', '0'], stderr=subprocess.DEVNULL)
    return jsonify({'ok': True, 'volume': vol})


@flask_app.route('/api/s3', methods=['POST'])
def api_s3():
    _sw_states['s2'] = False
    _pulse_switch('s3')
    state.reset_runtime_state(speak=True)
    return jsonify({'ok': True})


@flask_app.route('/api/camera_frame')
def api_camera_frame():
    if state.camera and state.camera.available:
        jpg = state.camera.get_frame_jpg()
        if jpg:
            return jsonify({'frame': jpg, 'source': 'local_camera', 'fresh': True})
    mode = state.current_mode() if state else None
    mode_name = mode.name if mode else ''
    candidates = []
    if mode_name == '辅助抓取模式' and state and state.rk3588:
        if state.rk3588.running:
            candidates.append(('rk3588_vector', state.rk3588.preview_output_path))
    elif mode_name == '导航识别模式' and state and state.rk3588_obstacle:
        if state.rk3588_obstacle.running:
            candidates.append(('rk3588_obstacle', state.rk3588_obstacle.preview_output_path))
    elif state and state.camera and state.camera.available:
        jpg = state.camera.get_frame_jpg()
        if jpg:
            return jsonify({'frame': jpg, 'source': 'local_camera', 'fresh': True})
    for source, path in candidates:
        data = _read_fresh_image_base64(path)
        if data:
            return jsonify({'frame': data, 'source': source, 'fresh': True})
    return jsonify({'frame': None, 'source': mode_name or 'unknown', 'fresh': False})


@flask_app.route('/api/obstacle_preview')
def api_obstacle_preview():
    path = state.rk3588_obstacle.preview_output_path if state and state.rk3588_obstacle else ''
    data = _read_fresh_image_base64(path)
    return jsonify({'frame': data, 'fresh': bool(data)})


def run_server(host='0.0.0.0', port=5000):
    global state
    state = create_app()
    _attach_runtime_hooks(state)
    flask_app.run(host=host, port=port, debug=False, threaded=True)
