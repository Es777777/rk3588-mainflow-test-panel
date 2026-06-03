import cv2
import threading
import base64
from app.config import get_config


def _check_camera_with_timeout(dev, timeout=3):
    result = [False]
    def _try():
        cap = cv2.VideoCapture(dev)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            result[0] = ret
    t = threading.Thread(target=_try, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result[0]


class Camera:
    def __init__(self):
        self._cap = None
        self._frame = None
        self._running = False
        self._available = False
        self._thread = None

    def check(self):
        cfg = get_config()
        dev = cfg.get('camera', {}).get('device_id', 0)
        self._available = _check_camera_with_timeout(dev)
        return self._available

    def start(self):
        if self._running:
            return
        cfg = get_config()
        dev = cfg.get('camera', {}).get('device_id', 0)
        self._cap = cv2.VideoCapture(dev)
        if not self._cap.isOpened():
            self._available = False
            return
        self._available = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        no_frame_count = 0
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                self._frame = frame
                no_frame_count = 0
            else:
                no_frame_count += 1
                if no_frame_count > 30:
                    self._available = False
                    break
            self._running and threading.Event().wait(0.03)

    def get_frame_jpg(self):
        if self._frame is None:
            return None
        _, jpg = cv2.imencode('.jpg', self._frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return base64.b64encode(jpg.tobytes()).decode()

    def get_frame_raw(self):
        return self._frame

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()

    @property
    def available(self):
        return self._available
