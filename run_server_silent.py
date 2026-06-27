from app.bootstrap import prepare_runtime

prepare_runtime()

from app.server import run_server
from app.main import create_app

original_create = create_app
def silent_create():
    state = original_create()
    state.tts._available = False
    state.speak_text = lambda t: (state.add_log(f'🔇 {t}'), True)[1]
    return state

import app.server as sm
sm.create_app = silent_create

run_server()
