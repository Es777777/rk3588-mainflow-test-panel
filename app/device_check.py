import cv2
import os
import re
import socket
import subprocess
import shutil
import tempfile
import time
import threading
from app.config import get_config


_V4L2_DEV_RE = re.compile(r'^video(\d+)$')
_camera_shared_state = None


def set_camera_state(state):
    global _camera_shared_state
    _camera_shared_state = state


def _probe_camera():
    cfg = get_config()
    dev = cfg.get('camera', {}).get('device_path', 0)
    if not dev or not os.path.exists(str(dev)):
        dev = cfg.get('camera', {}).get('device_id', 0)
    result = [False]
    def _try():
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                result[0] = True
    t = threading.Thread(target=_try, daemon=True)
    t.start()
    t.join(timeout=3)
    return result[0]

def _check_camera():
    if _camera_shared_state is not None:
        ok = _camera_shared_state.available
        if not ok:
            ok = _probe_camera()
        return {'ok': ok, 'detail': '摄像头正常' if ok else '未检测到摄像头'}
    ok = _probe_camera()
    return {'ok': ok, 'detail': '摄像头正常' if ok else '未检测到摄像头'}


def _check_microphone():
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        count = p.get_device_count()
        has_input = any(p.get_device_info_by_index(i).get('maxInputChannels', 0) > 0 for i in range(count))
        p.terminate()
        return {'ok': has_input, 'detail': '麦克风正常' if has_input else '未找到麦克风'}
    except ImportError:
        pass
    if shutil.which('arecord'):
        try:
            tmp = tempfile.mktemp(suffix='.wav')
            r = subprocess.run(['arecord', '-D', 'plughw:4,0', '-d', '1', '-f', 'S16_LE', '-c', '2', '-r', '16000', tmp],
                               timeout=3, stderr=subprocess.DEVNULL)
            ok = r.returncode == 0 and os.path.getsize(tmp) > 44
            return {'ok': ok, 'detail': '麦克风正常' if ok else '麦克风无响应'}
        except Exception:
            pass
    return {'ok': False, 'detail': '未检测到麦克风'}


def _check_speaker():
    if shutil.which('speaker-test'):
        try:
            r = subprocess.run(['speaker-test', '-l', '1', '-p', '1'], timeout=3, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok = r.returncode == 0
            return {'ok': ok, 'detail': '扬声器正常' if ok else '扬声器无响应'}
        except Exception:
            pass
    for cmd in ['aplay', 'paplay']:
        if shutil.which(cmd):
            return {'ok': True, 'detail': f'{cmd} 可用（需硬件验证）'}
    return {'ok': False, 'detail': '未检测到扬声器'}


def _check_network():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(3)
        s.connect(('8.8.8.8', 80))
        s.close()
        return {'ok': True, 'detail': '网络连通'}
    except OSError:
        return {'ok': False, 'detail': '网络不可用'}


def _check_bluetooth():
    if shutil.which('bluetoothctl'):
        try:
            out = subprocess.check_output(['bluetoothctl', 'show'], stderr=subprocess.DEVNULL, timeout=5)
            powered = b'Powered: yes' in out
            return {'ok': powered, 'detail': '蓝牙已开启' if powered else '蓝牙未开启'}
        except Exception:
            pass
    return {'ok': False, 'detail': '蓝牙不可用'}


def run_all():
    checks = {
        'camera': _check_camera(),
        'microphone': _check_microphone(),
        'speaker': _check_speaker(),
        'network': _check_network(),
        'bluetooth': _check_bluetooth(),
    }
    all_ok = all(v['ok'] for v in checks.values())
    return {'all_ok': all_ok, 'checks': checks}
