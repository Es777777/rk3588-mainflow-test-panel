import base64
import os
from pathlib import Path

import requests


class CloudVisionClient:
    def __init__(self, config=None, log=None):
        self._config = config or {}
        self._log = log or (lambda msg: None)
        self._enabled = bool(self._config.get('enabled', False))
        self._base_url = str(
            self._config.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        ).rstrip('/')
        api_key_file = str(self._config.get('api_key_file') or '').strip()
        self._api_key = str(
            self._config.get('api_key')
            or os.getenv(str(self._config.get('api_key_env', 'DASHSCOPE_API_KEY')) or 'DASHSCOPE_API_KEY')
            or self._read_key_file(api_key_file)
            or ''
        ).strip()
        self._model = str(self._config.get('model', 'qwen-vl-plus') or 'qwen-vl-plus').strip()
        self._timeout_s = float(self._config.get('timeout_s', 40) or 40)
        self._temperature = float(self._config.get('temperature', 0.2) or 0.2)
        self._max_tokens = int(self._config.get('max_tokens', 160) or 160)

    @property
    def enabled(self):
        return self._enabled

    def available(self):
        return bool(self._enabled and self._api_key and self._base_url and self._model)

    def _read_key_file(self, path):
        if not path:
            return ''
        try:
            return Path(path).expanduser().read_text(encoding='utf-8').strip()
        except Exception:
            return ''

    def analyze_image(self, image_path, prompt=''):
        image_path = str(image_path or '').strip()
        if not image_path or not self.available():
            return ''
        path = Path(image_path)
        if not path.exists():
            return ''

        system_prompt = (
            '你是眼镜端视觉理解助手。'
            '你只描述可见物体、位置关系和风险，不要复述用户问题。'
            '只输出简洁中文结论，不要编号，不要解释，不要 markdown。'
        )
        user_prompt = str(prompt or '我面前有什么').strip() or '我面前有什么'

        b64 = base64.b64encode(path.read_bytes()).decode('ascii')
        payload = {
            'model': self._model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                        {'type': 'text', 'text': user_prompt},
                    ],
                },
            ],
            'temperature': self._temperature,
            'max_tokens': self._max_tokens,
            'stream': False,
        }
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
        }
        response = requests.post(
            f'{self._base_url}/chat/completions',
            headers=headers,
            json=payload,
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        return self._normalize_reply((((data.get('choices') or [{}])[0]).get('message') or {}).get('content'))

    def _normalize_reply(self, reply):
        text = str(reply or '').strip()
        if not text:
            return ''
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ''
        primary = lines[0]
        if primary.startswith('问题：') or primary.startswith('问题:'):
            return ''
        for splitter in ('。', '！', '？'):
            if splitter in primary:
                primary = primary.split(splitter, 1)[0] + splitter
                break
        if len(primary) > 48:
            primary = primary[:48].rstrip('，,;； ') + '。'
        return primary
