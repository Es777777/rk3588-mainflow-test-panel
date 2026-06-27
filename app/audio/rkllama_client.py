import os
import signal
import subprocess
import tempfile
import threading
import time
import wave
import base64
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    from PIL import Image  # type: ignore[import-not-found]
except Exception:
    Image = None


class RkllamaClient:
    def __init__(self, config=None, log=None):
        self._config = config or {}
        self._log = log or (lambda msg: None)
        self._api_base = self._config.get('api_base', 'http://127.0.0.1:8080/v1').rstrip('/')
        self._stt_model = self._config.get('stt_model', 'whisper-tiny')
        self._tts_model = self._config.get('tts_model', 'zh_CN-huayan-medium')
        self._chat_model = self._config.get('chat_model', 'internvl3.5-1b')
        self._autostart = bool(self._config.get('autostart', False))
        self._startup_timeout_s = float(self._config.get('startup_timeout_s', 25))
        self._startup_log = self._config.get('startup_log')
        self._start_command = list(self._config.get('start_command') or [])
        self._extra_env = dict(self._config.get('env') or {})
        self._text_timeout_s = float(self._config.get('text_timeout_s', 90))
        self._vision_timeout_s = float(self._config.get('vision_timeout_s', 18))
        self._text_attempts = int(self._config.get('text_attempts', 2) or 2)
        self._vision_attempts = int(self._config.get('vision_attempts', 2) or 2)
        self._start_lock = threading.Lock()
        self._vision_warm_lock = threading.Lock()
        self._server_process = None
        self._vision_warmed = False

    def cleanup_duplicate_servers(self):
        pids = self._find_existing_server_pids()
        if self._port_responding():
            return pids
        if not pids:
            return []
        self._log(f'♻️ 仅在 rkllama 端口无响应时回收旧进程: pids={pids}')
        self._terminate_pids(pids)
        return []

    def available(self):
        try:
            response = requests.get(f'{self._api_base}/models', timeout=5)
            response.raise_for_status()
            return True
        except Exception:
            return False

    def ensure_available(self):
        if self.available():
            return True
        if not self._autostart or not self._start_command:
            return False
        with self._start_lock:
            if self.available():
                return True
            self._start_server_if_needed()
            deadline = time.time() + self._startup_timeout_s
            while time.time() < deadline:
                if self.available():
                    self._log('🧠 本地 rkllama 服务已就绪')
                    return True
                if self._server_process is not None and self._server_process.poll() is not None:
                    self._log(f'❌ rkllama 服务启动失败，退出码: {self._server_process.returncode}')
                    break
                time.sleep(1.0)
        return self.available()

    def _start_server_if_needed(self):
        existing_pids = self._find_existing_server_pids()
        if existing_pids:
            if self._port_responding():
                self._log(f'🧠 检测到已有 rkllama 服务进程: pids={existing_pids}')
                return
            self._log(f'♻️ 检测到未响应的 rkllama 旧进程，准备回收: pids={existing_pids}')
            self._terminate_pids(existing_pids)
        if self._server_process is not None and self._server_process.poll() is None:
            return
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        log_handle = None
        if self._startup_log:
            log_path = Path(self._startup_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_path, 'ab')
            stdout = log_handle
            stderr = subprocess.STDOUT
        self._log('🚀 正在拉起本地 rkllama 服务...')
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self._extra_env.items()})
        try:
            self._server_process = subprocess.Popen(
                self._start_command,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
        except Exception:
            if log_handle:
                log_handle.close()
            raise

    def _find_existing_server_pids(self):
        if not self._start_command:
            return []
        try:
            current_pid = os.getpid()
            output = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True, timeout=3)
            needle = ' '.join(str(part) for part in self._start_command)
            pids = []
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    pid_text, args = line.split(' ', 1)
                    pid = int(pid_text)
                except ValueError:
                    continue
                if pid == current_pid:
                    continue
                if needle in args:
                    pids.append(pid)
        except Exception:
            return []
        return pids

    def _port_responding(self):
        try:
            parsed = urlparse(self._api_base if '://' in self._api_base else f'http://{self._api_base}')
            host = parsed.hostname or '127.0.0.1'
            port = parsed.port or 80
            response = requests.get(f'http://{host}:{port}/v1/models', timeout=2)
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _terminate_pids(self, pids):
        alive = []
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                alive.append(pid)
            except ProcessLookupError:
                continue
            except Exception as exc:
                self._log(f'⚠️ 终止 rkllama 旧进程失败 pid={pid}: {exc}')
        if not alive:
            return
        deadline = time.time() + 3.0
        while time.time() < deadline:
            remaining = [pid for pid in alive if self._pid_alive(pid)]
            if not remaining:
                return
            time.sleep(0.2)
        for pid in alive:
            if self._pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass

    def _pid_alive(self, pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except Exception:
            return True

    def transcribe(self, audio_path, language='zh'):
        with open(audio_path, 'rb') as handle:
            response = requests.post(
                f'{self._api_base}/audio/transcriptions',
                files={'file': (os.path.basename(audio_path), handle, 'audio/wav')},
                data={'model': self._stt_model, 'language': language, 'response_format': 'json'},
                timeout=300,
            )
        response.raise_for_status()
        data = response.json()
        return data.get('text', '').strip(), data

    def chat(self, prompt, system_prompt):
        payload = {
            'model': self._chat_model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
            ],
            'stream': False,
        }
        return self._chat_completion(
            payload,
            require_non_empty=True,
            request_name='text_chat',
            attempts=self._text_attempts,
            timeout_s=self._text_timeout_s,
        )

    def chat_with_image(self, prompt, system_prompt, image_path=None, attempts=None, timeout_s=None, warm=True):
        if warm:
            self._ensure_vision_warm(image_path=image_path)
        user_content = [{'type': 'text', 'text': prompt}]
        if image_path:
            image_path = Path(image_path)
            if image_path.exists():
                mime, payload_bytes = self._prepare_image_payload(image_path)
                b64 = base64.b64encode(payload_bytes).decode('ascii')
                user_content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:{mime};base64,{b64}',
                    },
                })
        payload = {
            'model': self._chat_model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
            ],
            'stream': False,
        }
        return self._chat_completion(
            payload,
            require_non_empty=True,
            request_name='vision_chat',
            attempts=int(attempts or self._vision_attempts),
            timeout_s=float(timeout_s or self._vision_timeout_s),
        )

    def warmup_vision(self, image_path=None):
        self._ensure_vision_warm(image_path=image_path, force=True)

    def _chat_completion(self, payload, require_non_empty=False, request_name='chat', attempts=3, timeout_s=300):
        attempts = max(1, int(attempts or 1))
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(
                    f'{self._api_base}/chat/completions',
                    json=payload,
                    timeout=timeout_s,
                )
                response.raise_for_status()
                data = response.json()
                content = self._extract_message_content(data).strip()
                if require_non_empty and not content:
                    raise ValueError('empty assistant content')
                return content, data
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    raise
                delay_s = 1.0 * attempt
                self._log(f'⚠️ rkllama {request_name} 第{attempt}次失败，{delay_s:.1f}s后重试: {exc}')
                time.sleep(delay_s)
        raise last_error

    def _extract_message_content(self, data):
        choices = data.get('choices') if isinstance(data, dict) else None
        if not choices:
            return ''
        message = choices[0].get('message') or {}
        return str(message.get('content') or '')

    def _ensure_vision_warm(self, image_path=None, force=False):
        if self._vision_warmed and not force:
            return
        with self._vision_warm_lock:
            if self._vision_warmed and not force:
                return
            payload = self._build_vision_warmup_payload(image_path=image_path)
            try:
                self._chat_completion(payload, require_non_empty=False, request_name='vision_warmup')
                self._vision_warmed = True
                self._log('🧠 多模态视觉通路预热完成')
            except Exception as exc:
                self._log(f'⚠️ 多模态视觉通路预热失败: {exc}')

    def _build_vision_warmup_payload(self, image_path=None):
        user_content = [{'type': 'text', 'text': '请只回答“好的”。'}]
        if image_path:
            image_path = Path(image_path)
            if image_path.exists():
                mime, payload_bytes = self._prepare_image_payload(image_path)
                b64 = base64.b64encode(payload_bytes).decode('ascii')
                user_content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:{mime};base64,{b64}',
                    },
                })
        return {
            'model': self._chat_model,
            'messages': [
                {'role': 'system', 'content': '你是视觉助手，只输出两个字：好的'},
                {'role': 'user', 'content': user_content},
            ],
            'stream': False,
        }

    def _prepare_image_payload(self, image_path):
        suffix = image_path.suffix.lower()
        default_mime = 'image/png' if suffix == '.png' else 'image/jpeg'
        raw = image_path.read_bytes()
        if Image is None:
            return default_mime, raw
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                max_side = max(img.size)
                if max_side > 1280:
                    scale = 1280.0 / float(max_side)
                    resized = (
                        max(1, int(round(img.size[0] * scale))),
                        max(1, int(round(img.size[1] * scale))),
                    )
                    img = img.resize(resized)
                tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                tmp_path = Path(tmp.name)
                tmp.close()
                try:
                    img.save(tmp_path, format='JPEG', quality=85, optimize=True)
                    return 'image/jpeg', tmp_path.read_bytes()
                finally:
                    tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            self._log(f'⚠️ 图像预处理失败，回退原图上传: {exc}')
            return default_mime, raw

    def synthesize(self, text, output_path):
        response = requests.post(
            f'{self._api_base}/audio/speech',
            json={
                'model': self._tts_model,
                'input': text,
                'response_format': 'wav',
                'speed': 1.0,
            },
            timeout=300,
        )
        response.raise_for_status()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return output_path

    def convert_to_glasses_wav(self, src_path, dst_path, volume=0.7):
        subprocess.run(
            [
                'ffmpeg',
                '-y',
                '-i',
                str(src_path),
                '-ar',
                '16000',
                '-ac',
                '1',
                '-sample_fmt',
                's16',
                '-filter:a',
                f'volume={volume}',
                str(dst_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return Path(dst_path)

    def speak_via_glasses(self, glasses_bridge, text, out_dir):
        if not glasses_bridge or not glasses_bridge.available or not text.strip():
            return False
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / 'tts_reply.wav'
        play_path = out_dir / 'tts_reply_16k.wav'
        self.synthesize(text, raw_path)
        self.convert_to_glasses_wav(raw_path, play_path)
        return glasses_bridge.play_wav(str(play_path))
