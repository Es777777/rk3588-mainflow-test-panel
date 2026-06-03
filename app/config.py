import json
import os

_CONFIG = None


def load_config(path=None):
    global _CONFIG
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    with open(path) as f:
        _CONFIG = json.load(f)
    return _CONFIG


def get_config():
    global _CONFIG
    if _CONFIG is None:
        load_config()
    return _CONFIG
