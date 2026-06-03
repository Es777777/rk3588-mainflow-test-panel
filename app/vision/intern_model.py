class InternModel:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._available = False

    def check(self):
        try:
            from transformers import AutoModel, AutoTokenizer
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def load(self):
        if not self._available:
            return False
        try:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-chat-7b', trust_remote_code=True)
            self._model = AutoModel.from_pretrained('internlm/internlm-chat-7b', trust_remote_code=True).cuda()
            self._available = True
            return True
        except Exception:
            self._available = False
            return False

    def chat(self, prompt):
        if not self._available:
            return 'Intern 模型未加载'
        try:
            response, _ = self._model.chat(self._tokenizer, prompt, history=[])
            return response
        except Exception as e:
            return f'模型调用失败: {e}'

    @property
    def available(self):
        return self._available
