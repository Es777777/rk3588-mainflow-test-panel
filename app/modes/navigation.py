from app.modes.base import BaseMode


class NavigationMode(BaseMode):
    name = '导航模式'

    def __init__(self, state):
        super().__init__(state)
        self.destination = None

    def on_enter(self):
        if self.state.tts.available:
            self.state.tts.speak('当前为导航模式')
        self.state.add_log('🧭 导航模式启动 — 导航模块待接入')
