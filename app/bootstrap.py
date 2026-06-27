import os
import sys
import types


def prepare_runtime():
    root = os.path.dirname(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    if 'torchvision' in sys.modules:
        return

    tv = types.ModuleType('torchvision')
    tv.__version__ = '0.27.1'
    ops = types.ModuleType('torchvision.ops')
    tv.ops = ops
    sys.modules['torchvision'] = tv
    sys.modules['torchvision.ops'] = ops
