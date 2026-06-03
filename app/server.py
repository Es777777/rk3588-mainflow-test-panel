import os
from flask import Flask, jsonify, request, send_from_directory
from app.main import create_app

state = None
_sw_states = {'s1': False, 's2': False, 's3': False}
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
os.makedirs(static_dir, exist_ok=True)
flask_app = Flask(__name__, static_folder=static_dir)


@flask_app.route('/')
def index():
    return send_from_directory(static_dir, 'index.html')


@flask_app.route('/api/status')
def api_status():
    mode = state.current_mode()
    return jsonify({
        'mode': mode.name if mode else 'unknown',
        'mode_index': state.mode_index,
        'device_status': state.device_status,
        'stt_text': state.stt_text,
        'target': state.target,
        'detection': state.detection,
        'obstacles': state.obstacles[-5:] if state.obstacles else [],
        'logs': state.logs[-100:],
        'switches': dict(_sw_states),
        'camera_available': state.camera.available,
        'yolo_available': state.yolo.available if state.yolo else False,
        'stt_available': state.stt.available if state.stt else False,
        'tts_available': state.tts.available if state.tts else False,
    })


@flask_app.route('/api/s1', methods=['POST'])
def api_s1():
    _sw_states['s1'] = True
    state.cycle_mode()
    return jsonify({'ok': True})


@flask_app.route('/api/s2', methods=['POST'])
def api_s2():
    data = request.get_json()
    action = data.get('action', 'press')
    mode = state.current_mode()
    if action == 'press':
        _sw_states['s2'] = True
        if mode:
            mode.on_s2_press()
    elif action == 'release':
        _sw_states['s2'] = False
        if mode:
            mode.on_s2_release()
    return jsonify({'ok': True})


@flask_app.route('/api/s3', methods=['POST'])
def api_s3():
    _sw_states['s3'] = True
    state.set_mode(0)
    state.stt_text = ''
    state.target = ''
    state.detection = {}
    state.obstacles = []
    state.add_log('🔄 已重置')
    if state.tts.available:
        state.tts.speak('已重置')
    return jsonify({'ok': True})


@flask_app.route('/api/camera_frame')
def api_camera_frame():
    if state.camera and state.camera.available:
        jpg = state.camera.get_frame_jpg()
        if jpg:
            return jsonify({'frame': jpg})
    return jsonify({'frame': None})


def run_server(host='0.0.0.0', port=5000):
    global state
    state = create_app()
    flask_app.run(host=host, port=port, debug=False, threaded=True)
