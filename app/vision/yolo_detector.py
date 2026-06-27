import threading
import time


class YOLODetector:
    def __init__(self):
        self._model = None
        self._available = False
        self._camera = None
        self._cached_results = []
        self._cache_lock = threading.Lock()
        self._bg_running = False
        self._bg_thread = None

    def check(self):
        try:
            from ultralytics import YOLO
            result = [None]
            def _load():
                try:
                    result[0] = YOLO('yolov8n.pt')
                except Exception:
                    pass
            t = threading.Thread(target=_load, daemon=True)
            t.start()
            t.join(timeout=10)
            if result[0] is not None:
                self._model = result[0]
                self._available = True
        except ImportError:
            self._available = False
        return self._available

    def start_background(self, camera):
        if not self._available or self._bg_running:
            return
        self._camera = camera
        self._bg_running = True
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()

    def stop_background(self):
        self._bg_running = False
        if self._bg_thread:
            self._bg_thread.join(timeout=2)
            self._bg_thread = None

    def _bg_loop(self):
        interval = 0.2
        while self._bg_running:
            if self._camera and self._camera.available:
                frame = self._camera.get_frame_raw()
                if frame is not None:
                    try:
                        results = self._model(frame)
                        dets = []
                        for r in results:
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                label = r.names[cls_id]
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                dets.append({
                                    'label': label,
                                    'confidence': round(conf, 2),
                                    'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                                    'center': {'x': (x1 + x2) // 2, 'y': (y1 + y2) // 2},
                                })
                        with self._cache_lock:
                            self._cached_results = dets
                    except Exception:
                        pass
            time.sleep(interval)

    def detect(self, frame, target=None):
        if not self._available or self._model is None:
            return []
        if self._bg_running:
            with self._cache_lock:
                results = list(self._cached_results)
            if target:
                return [d for d in results if d['label'] == target]
            return results
        try:
            results = self._model(frame)
            dets = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = r.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    dets.append({
                        'label': label,
                        'confidence': round(conf, 2),
                        'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'center': {'x': (x1 + x2) // 2, 'y': (y1 + y2) // 2},
                    })
            return dets
        except Exception:
            return []

    @property
    def available(self):
        return self._available
