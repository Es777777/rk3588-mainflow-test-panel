import json
import os

_CONFIG = None
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


def resolve_project_path(path):
    if not path:
        return ''
    expanded = os.path.expanduser(str(path))
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(ROOT_DIR, expanded))


def load_config(path=None):
    global _CONFIG
    if path is None:
        path = os.path.join(ROOT_DIR, 'config.json')
    with open(path, encoding='utf-8') as f:
        _CONFIG = json.load(f)
    return _CONFIG


def get_config():
    global _CONFIG
    if _CONFIG is None:
        load_config()
    return _CONFIG
