class STT:
    def __init__(self):
        self._available = False

    def check(self):
        try:
            import whisper
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    @property
    def available(self):
        return self._available

    def transcribe(self, audio_path):
        if not self._available:
            return ''
        try:
            import whisper
            model = whisper.load_model('base')
            result = model.transcribe(audio_path)
            return result.get('text', '').strip()
        except Exception:
            return ''
