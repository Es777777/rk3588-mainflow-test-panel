class InternModel:
    def __init__(self, rkllama_client=None, log=None):
        self._rkllama = rkllama_client
        self._log = log or (lambda msg: None)
        self._available = False

    def check(self):
        self._available = bool(self._rkllama and self._rkllama.ensure_available())
        return self._available

    def analyze_image(self, image_path, prompt=''):
        image_path = str(image_path or '').strip()
        if not image_path:
            return ''
        if not self.check():
            return ''
        analysis_prompt = (
            prompt
            or '请用简体中文简短描述这张图里最重要的物体、位置关系和潜在风险，控制在40字以内。'
        )
        system_prompt = (
            '你是眼镜端视觉理解助手。'
            '你只描述可见物体、位置关系和风险，不做身份识别，不输出拒答模板。'
            '只输出简洁中文结论，不要编号，不要解释，不要 markdown。'
        )
        try:
            reply, _ = self._rkllama.chat_with_image(
                prompt=analysis_prompt,
                system_prompt=system_prompt,
                image_path=image_path,
                attempts=1,
                warm=False,
            )
            return self._normalize_reply(reply)
        except Exception as exc:
            self._log(f'⚠️ Intern 画面分析失败: {exc}')
            self._available = False
            return ''

    def _normalize_reply(self, reply):
        text = str(reply or '').strip()
        if not text:
            return ''
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ''
        primary = lines[0]
        for splitter in ('。', '！', '？'):
            if splitter in primary:
                primary = primary.split(splitter, 1)[0] + splitter
                break
        if len(primary) > 48:
            primary = primary[:48].rstrip('，,;； ') + '。'
        return primary

    @property
    def available(self):
        return self._available
