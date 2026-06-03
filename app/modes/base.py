class BaseMode:
    name = 'base'

    def __init__(self, state):
        self.state = state

    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def on_s2_press(self):
        pass

    def on_s2_release(self):
        pass

    def update(self):
        pass
