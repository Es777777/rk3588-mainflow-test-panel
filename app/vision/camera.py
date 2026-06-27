import cv2
import os
import re
import threading
import base64
import time
from app.config import get_config, resolve_project_path


_V4L2_DEV_RE = re.compile(r'^video(\d+)$')
_FRAME_MAX_AGE = 3.0


def _api_to_backend(api):
    value = api.lower()
    mapping = {
        "auto": cv2.CAP_ANY,
        "v4l2": cv2.CAP_V4L2,
    }
    return mapping.get(value, cv2.CAP_ANY)


def _pixel_format_to_fourcc(pixel_format):
    value = str(pixel_format or '').strip().upper()
    if len(value) == 4:
        return cv2.VideoWriter_fourcc(*value)
    return None


def _probe_device(index, api):
    cap = cv2.VideoCapture(index, api)
    if cap.isOpened():
        cap.release()
        return True
    return False


def _scan_v4l2_devices():
    devices = []
    for entry in os.listdir('/dev'):
        m = _V4L2_DEV_RE.match(entry)
        if m:
            devices.append(int(m.group(1)))
    return sorted(devices)


def _auto_detect_device(api):
    for idx in _scan_v4l2_devices():
        try:
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None and frame.size > 0:
                    return idx
        except Exception:
            continue
    return None


class Camera:
    def __init__(self, log=None):
        self._cap = None
        self._frame = None
        self._frame_ts = 0
        self._lock = threading.Lock()
        self._running = False
        self._available = False
        self._thread = None
        self._reconnect_attempts = 0
        self._active_source = None
        self._last_jpg = None
        self._last_jpg_ts = 0
        self._log = log or (lambda msg: None)
        self._last_failure_log_ts = 0.0

    def _source(self, cam_cfg):
        path = cam_cfg.get('device_path', '')
        api = _api_to_backend(cam_cfg.get('api', 'v4l2'))
        if path and os.path.exists(path):
            return path
        preferred = cam_cfg.get('device_id', 0)
        if _probe_device(preferred, api):
            return preferred
        detected = _auto_detect_device(api)
        if detected is not None:
            return detected
        return preferred

    def _source_for_retry(self, api):
        cfg = get_config()
        cam_cfg = cfg.get('camera', {})
        preferred_path = cam_cfg.get('device_path', '')
        if preferred_path and os.path.exists(preferred_path):
            return preferred_path
        preferred = cam_cfg.get('device_id', 0)
        if _probe_device(preferred, api):
            return preferred
        return _auto_detect_device(api)

    def check(self):
        cfg = get_config()
        cam_cfg = cfg.get('camera', {})
        dev = self._source(cam_cfg)
        api = _api_to_backend(cam_cfg.get('api', 'v4l2'))
        cap = cv2.VideoCapture(dev, api)
        if cap.isOpened():
            cap.release()
            self._available = True
        else:
            self._available = False
        return self._available

    def _open(self, dev, api):
        if self._cap:
            self._cap.release()
            self._cap = None
        self._cap = cv2.VideoCapture(dev, api)
        opened = self._cap and self._cap.isOpened()
        self._active_source = dev if opened else None
        if not opened:
            self._rate_limited_log(f'⚠️ 摄像头打开失败: source={dev} api={api}')
        return opened

    def _apply_capture_settings(self, cam_cfg):
        if not self._cap:
            return
        pixel_format = _pixel_format_to_fourcc(cam_cfg.get('pixel_format', 'MJPG'))
        if pixel_format is not None:
            self._cap.set(cv2.CAP_PROP_FOURCC, pixel_format)
        buffer_size = int(cam_cfg.get('buffer_size', 1) or 1)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, buffer_size))
        except Exception:
            pass
        w = cam_cfg.get('frame_width', 0)
        h = cam_cfg.get('frame_height', 0)
        fps = cam_cfg.get('fps', 0)
        if w > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, fps)

    def _capture_diagnostics(self, cap=None):
        cap = cap or self._cap
        if not cap:
            return {}
        try:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            pixel_format = ''.join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip('\x00') or 'unknown'
        except Exception:
            pixel_format = 'unknown'
        try:
            return {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                'fps': round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 2),
                'pixel_format': pixel_format,
                'buffer_size': int(cap.get(cv2.CAP_PROP_BUFFERSIZE) or 0),
            }
        except Exception:
            return {'pixel_format': pixel_format}

    def diagnostics(self):
        with self._lock:
            data = self._capture_diagnostics()
            data.update({
                'available': self._available,
                'active_source': self._active_source,
                'frame_age_ms': round(max(0.0, time.time() - self._frame_ts) * 1000.0, 1) if self._frame_ts else None,
                'reconnect_attempts': self._reconnect_attempts,
            })
            return data

    def _format_diag(self, data):
        return (
            f"source={data.get('active_source')} "
            f"fmt={data.get('pixel_format')} "
            f"size={data.get('width')}x{data.get('height')} "
            f"fps={data.get('fps')} "
            f"buffer={data.get('buffer_size')} "
            f"age_ms={data.get('frame_age_ms')}"
        )

    def _rate_limited_log(self, message, min_interval_s=5.0):
        now = time.time()
        if now - self._last_failure_log_ts >= min_interval_s:
            self._last_failure_log_ts = now
            self._log(message)

    def active_source(self):
        with self._lock:
            return self._active_source

    def start(self):
        if self._running:
            return True
        cfg = get_config()
        cam_cfg = cfg.get('camera', {})
        api = _api_to_backend(cam_cfg.get('api', 'v4l2'))
        dev = self._source(cam_cfg)
        if not self._open(dev, api):
            self._available = False
            return False
        self._apply_capture_settings(cam_cfg)
        diag = self.diagnostics()
        self._log(f'📷 本地相机启动参数: {self._format_diag(diag)}')
        self._available = True
        self._reconnect_attempts = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _reconnect(self, api):
        dev = self._source_for_retry(api)
        cfg = get_config()
        cam_cfg = cfg.get('camera', {})
        if dev is None:
            with self._lock:
                self._available = False
            return False
        self._open(dev, api)
        if self._cap and self._cap.isOpened():
            self._apply_capture_settings(cam_cfg)
            with self._lock:
                self._available = True
                self._reconnect_attempts = 0
            diag = self.diagnostics()
            self._log(f'📷 摄像头重连成功: {self._format_diag(diag)}')
            return True
        with self._lock:
            self._available = False
        self._rate_limited_log(f'⚠️ 摄像头重连失败: source={dev}')
        return False

    def _loop(self):
        cfg = get_config()
        api = _api_to_backend(cfg.get('camera', {}).get('api', 'v4l2'))
        retry_interval = float(cfg.get('camera', {}).get('retry_interval_s', 3))
        while self._running:
            if not self._running:
                break
            cap = self._cap
            if cap is None or not cap.isOpened():
                if not self._running:
                    break
                self._reconnect(api)
                self._reconnect_attempts += 1
                threading.Event().wait(max(1.0, retry_interval))
                continue
            try:
                ret, frame = cap.read()
            except Exception:
                ret, frame = False, None
                self._cap = None
            with self._lock:
                if ret and frame is not None:
                    ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    self._frame = frame
                    self._frame_ts = time.time()
                    self._last_jpg = base64.b64encode(jpg.tobytes()).decode() if ok else None
                    self._last_jpg_ts = self._frame_ts if ok else 0
                    self._available = True
                    self._reconnect_attempts = 0
                else:
                    self._frame = None
                    self._frame_ts = 0
                    self._last_jpg = None
                    self._last_jpg_ts = 0
                    self._available = False
                    if not self._running:
                        break
                    self._rate_limited_log('⚠️ 摄像头读帧失败，正在尝试重连')
                    self._reconnect(api)
                    self._reconnect_attempts += 1
                    threading.Event().wait(max(0.5, retry_interval))

    def get_frame_jpg(self):
        with self._lock:
            if self._last_jpg is None:
                return None
            if time.time() - self._last_jpg_ts > _FRAME_MAX_AGE:
                return None
            return self._last_jpg

    def get_frame_raw(self):
        with self._lock:
            if self._frame is None:
                return None
            if time.time() - self._frame_ts > _FRAME_MAX_AGE:
                return None
            return self._frame

    def open_once(self):
        cfg = get_config()
        cam_cfg = cfg.get('camera', {})
        api = _api_to_backend(cam_cfg.get('api', 'v4l2'))
        dev = self._source(cam_cfg)
        cap = cv2.VideoCapture(dev, api)
        if not cap.isOpened():
            self._rate_limited_log(f'⚠️ 单帧抓取失败，无法打开摄像头: source={dev} api={api}')
            return
        pixel_format = _pixel_format_to_fourcc(cam_cfg.get('pixel_format', 'MJPG'))
        if pixel_format is not None:
            cap.set(cv2.CAP_PROP_FOURCC, pixel_format)
        w = cam_cfg.get('frame_width', 0)
        h = cam_cfg.get('frame_height', 0)
        fps = cam_cfg.get('fps', 0)
        if w > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps > 0:
            cap.set(cv2.CAP_PROP_FPS, fps)
        time.sleep(0.3)
        ret, frame = cap.read()
        if ret and frame is not None:
            from pathlib import Path
            out_dir = Path(resolve_project_path('runtime_outputs/agent'))
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / 'latest_scene.jpg'
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            diag = self._capture_diagnostics(cap)
            diag['active_source'] = dev
            diag['frame_age_ms'] = 0.0
            self._log(f'📷 单帧预览已更新: {self._format_diag(diag)}')
        cap.release()

    def stop(self):
        self._running = False
        cap = self._cap
        self._cap = None
        if cap:
            cap.release()
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        with self._lock:
            self._frame = None
            self._frame_ts = 0
            self._last_jpg = None
            self._last_jpg_ts = 0
            self._available = False
            self._active_source = None

    @property
    def available(self):
        with self._lock:
            return self._available
