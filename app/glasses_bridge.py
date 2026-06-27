import os
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path


class GlassesBridge:
    def __init__(self, config=None, log=None):
        self._config = config or {}
        self._log = log or (lambda msg: None)
        self._device = None
        self._available = False
        self._connected = False
        self._pressed = False
        self._recording = False
        self._record_pcm = bytearray()
        self._last_wav_path = ''
        self._last_error = ''
        self._port = ''
        self._event_handlers = []
        self._record_lock = threading.Lock()
        self._play_lock = threading.Condition()
        self._playing = False
        self._last_play_done_ts = 0.0
        self._play_chunk_delay_s = float(self._config.get('play_chunk_delay_s', 0.002) or 0.0)
        self._play_start_delay_s = float(self._config.get('play_start_delay_s', 0.1) or 0.0)

    @property
    def enabled(self):
        return bool(self._config.get('enabled', False))

    @property
    def available(self):
        return self._available

    @property
    def connected(self):
        return self._connected

    @property
    def port(self):
        return self._port

    @property
    def last_wav_path(self):
        return self._last_wav_path

    @property
    def last_error(self):
        return self._last_error

    def add_event_handler(self, callback):
        self._event_handlers.append(callback)

    def start(self):
        if not self.enabled:
            return
        try:
            device = self._open_device()
            if device is None:
                return
            self._device = device
            self._device.on_audio(self._on_audio)
            self._device.on_event(self._on_event)
            button_name = str(self._config.get('button_name', 'EXTRA')).upper()
            self._device.on_button(button_name, 'down', lambda event: self._handle_button('down', event))
            self._device.on_button(button_name, 'up', lambda event: self._handle_button('up', event))
            self._device.on_button('MODE', 'short', lambda event: self._handle_button('mode_short', event))
            self._device.start()
            self._device.get_status()
            self._available = True
            self._connected = True
            self._port = self._device.port or ''
            self._log(f'🎛️ Glasses SDK 已连接: {self._port or "auto"}')
        except Exception as exc:
            self._last_error = str(exc)
            self._available = False
            self._connected = False
            self._log(f'⚠️ Glasses SDK 连接失败: {exc}')

    def stop(self):
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
        self._connected = False
        self._pressed = False
        with self._record_lock:
            self._recording = False
            self._record_pcm.clear()

    def start_recording(self):
        if not self._device:
            return False
        with self._record_lock:
            self._record_pcm.clear()
            self._recording = True
            self._last_wav_path = ''
        try:
            self._device.start_recording()
            return True
        except Exception as exc:
            with self._record_lock:
                self._recording = False
            self._last_error = str(exc)
            self._log(f'❌ Glasses SDK 开始录音失败: {exc}')
            return False

    def stop_recording(self, wait_timeout=8.0):
        if not self._device:
            return None
        try:
            self._device.stop_recording()
        except Exception as exc:
            self._last_error = str(exc)
            self._log(f'❌ Glasses SDK 停止录音失败: {exc}')
            return None

        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            with self._record_lock:
                if not self._recording and self._last_wav_path:
                    return self._last_wav_path
            time.sleep(0.05)
        return self._last_wav_path or None

    def play_wav(self, wav_path):
        if not self._device or not wav_path:
            return False
        try:
            with self._play_lock:
                self._playing = True
                self._last_play_done_ts = 0.0
            duration_s = self._wav_duration_s(wav_path)
            self._log(
                f'🎧 眼镜播放参数: duration={duration_s:.2f}s '
                f'chunk_delay={self._play_chunk_delay_s:.3f}s start_delay={self._play_start_delay_s:.3f}s'
            )
            self._device.play_wav(
                wav_path,
                chunk_delay=self._play_chunk_delay_s,
                start_delay=self._play_start_delay_s,
            )
            deadline = time.time() + max(4.0, duration_s + 2.0)
            with self._play_lock:
                finished = self._play_lock.wait_for(
                    lambda: (not self._playing) and self._last_play_done_ts > 0.0,
                    timeout=max(0.0, deadline - time.time()),
                )
            if finished:
                self._log('✅ Glasses SDK 播放完成')
                return True
            self._last_error = 'glasses play timeout'
            self._log('⚠️ Glasses SDK 播放超时，未收到 PLAY_DONE')
            with self._play_lock:
                self._playing = False
                self._play_lock.notify_all()
            return False
        except Exception as exc:
            self._last_error = str(exc)
            self._log(f'❌ Glasses SDK 播放录音失败: {exc}')
            with self._play_lock:
                self._playing = False
                self._play_lock.notify_all()
            return False

    def _wav_duration_s(self, wav_path):
        try:
            with wave.open(str(wav_path), 'rb') as wav_file:
                rate = float(wav_file.getframerate() or 0)
                if rate <= 0:
                    return 0.0
                return float(wav_file.getnframes()) / rate
        except Exception:
            return 0.0

    def _open_device(self):
        host_dir = self._host_dir()
        if not host_dir:
            self._last_error = 'Glasses SDK host_controller 路径不存在'
            self._log(f'⚠️ {self._last_error}')
            return None
        if host_dir not in sys.path:
            sys.path.insert(0, host_dir)
        from glasses_device import GlassesDevice  # type: ignore

        port = str(self._config.get('port', '')).strip() or None
        baudrate = int(self._config.get('baudrate', 115200))
        reconnect = bool(self._config.get('reconnect', True))
        return GlassesDevice.open(port=port, baud=baudrate, reconnect=reconnect)

    def _host_dir(self):
        base_dir = str(self._config.get('host_controller_dir', '')).strip()
        if not base_dir:
            return ''
        path = os.path.abspath(os.path.expanduser(base_dir))
        return path if os.path.isdir(path) else ''

    def _recordings_dir(self):
        path = str(self._config.get('recordings_dir', 'runtime_outputs/glasses_recordings')).strip()
        resolved = os.path.abspath(os.path.expanduser(path))
        os.makedirs(resolved, exist_ok=True)
        return resolved

    def _write_pcm_to_wav(self, pcm):
        host_dir = self._host_dir()
        if host_dir and host_dir not in sys.path:
            sys.path.insert(0, host_dir)
        from audio_utils import write_pcm_to_wav  # type: ignore

        path = Path(self._recordings_dir()) / f"talk_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        write_pcm_to_wav(path, pcm)
        return str(path)

    def _on_audio(self, chunk):
        with self._record_lock:
            if self._recording:
                self._record_pcm.extend(chunk)

    def _on_event(self, event):
        kind = getattr(event, 'kind', '')
        data = getattr(event, 'data', None)
        if kind == 'record_start':
            with self._record_lock:
                self._recording = True
                self._record_pcm.clear()
            self._log('🎙️ Glasses SDK 录音开始')
        elif kind == 'record_end':
            pcm = b''
            with self._record_lock:
                self._recording = False
                pcm = bytes(self._record_pcm)
                self._record_pcm.clear()
            if pcm:
                self._last_wav_path = self._write_pcm_to_wav(pcm)
                self._log(f'💾 Glasses SDK 已保存录音: {self._last_wav_path}')
            else:
                self._last_wav_path = ''
                self._log('⚠️ Glasses SDK 录音结束，但未收到音频数据')
        elif kind == 'disconnect':
            self._connected = False
            self._pressed = False
            with self._record_lock:
                self._recording = False
                self._record_pcm.clear()
            with self._play_lock:
                self._playing = False
                self._play_lock.notify_all()
            self._last_wav_path = ''
            self._last_error = str(data or 'disconnect')
            self._log(f'⚠️ Glasses SDK 已断开: {self._last_error}')
        elif kind == 'hello':
            self._connected = True
        elif kind == 'status':
            self._connected = True
            try:
                playing = bool((data or {}).get('playing'))
            except Exception:
                playing = self._playing
            with self._play_lock:
                self._playing = playing
                if not playing:
                    self._play_lock.notify_all()
        elif kind == 'play_done':
            with self._play_lock:
                self._playing = False
                self._last_play_done_ts = time.time()
                self._play_lock.notify_all()

        for callback in self._event_handlers:
            try:
                callback(kind, data)
            except Exception:
                pass

    def _handle_button(self, action, event):
        self._pressed = action == 'down'
        for callback in self._event_handlers:
            threading.Thread(
                target=self._safe_event_callback,
                args=(callback, f'button_{action}', event),
                daemon=True,
            ).start()

    def _safe_event_callback(self, callback, kind, data):
        try:
            callback(kind, data)
        except Exception:
            pass
