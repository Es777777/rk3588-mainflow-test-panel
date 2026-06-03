import threading


class YOLODetector:
    def __init__(self):
        self._model = None
        self._available = False

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

    def detect(self, frame, target=None):
        if not self._available or self._model is None:
            return []
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
