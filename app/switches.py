from app.config import get_config


class SwitchManager:
    def __init__(self, on_s1=None, on_s2_on=None, on_s2_off=None, on_s3=None):
        self.s1_state = False
        self.s2_state = False
        self.s3_state = False
        self._on_s1 = on_s1
        self._on_s2_on = on_s2_on
        self._on_s2_off = on_s2_off
        self._on_s3 = on_s3

    def press_s1(self):
        self.s1_state = True
        if self._on_s1:
            self._on_s1()

    def press_s2(self):
        self.s2_state = True
        if self._on_s2_on:
            self._on_s2_on()

    def release_s2(self):
        self.s2_state = False
        if self._on_s2_off:
            self._on_s2_off()

    def press_s3(self):
        self.s3_state = not self.s3_state
        if self._on_s3:
            self._on_s3()

    def get_states(self):
        return {
            's1': self.s1_state,
            's2': self.s2_state,
            's3': self.s3_state,
        }
