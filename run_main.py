from app.bootstrap import prepare_runtime

prepare_runtime()

from app.main import create_app
from app.server import flask_app
import app.server as sm
import app.main as mm

# Suppress audio at class level
mm.AppState.speak_text = lambda self, text: (self.add_log(f'🔇 {text}'), True)[1]

state = create_app()
state.tts._available = False
sm.state = state

# Re-mute after create_app (which calls _init_audio)
import subprocess
subprocess.run(['amixer', '-c', '4', 'sset', 'Headphone', '0'], stderr=subprocess.DEVNULL)
subprocess.run(['amixer', '-c', '4', 'sset', 'Speaker', 'off'], stderr=subprocess.DEVNULL)

flask_app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
