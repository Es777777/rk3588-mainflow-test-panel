import os, sys, threading, time
sys.path.insert(0, os.path.dirname(__file__))

from app.server import flask_app, state as app_state
from app.main import create_app

# Suppress audio
state = create_app()
state.tts._available = False
state._accept_glasses_buttons = False

def _silent_speak(text):
    state.add_log(f'🔇 (语音已静音): {text}')
state.speak_text = _silent_speak

# Set initial mode to assist grab without speaking
state.set_mode(0)

# Start Flask server
print('\n🌐 Starting Flask server at http://localhost:5000')
print('   Camera feed + vector data available on web UI')
print('   Press Ctrl+C to stop\n')

flask_app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
