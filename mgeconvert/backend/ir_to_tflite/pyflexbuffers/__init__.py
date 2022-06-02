import ctypes
import json
import os

libflatbuffer = os.path.join(os.path.dirname(__file__), "lib", "libflatbuffers.so.1")
ctypes.cdll.LoadLibrary(libflatbuffer)

# pylint: disable=import-error,no-name-in-module,wrong-import-position
from .fbconverter import (  # isort:skip
    from_json,
    to_json,
)


def loads(s):
    return json.loads(to_json(bytes(s)))


def dumps(obj, *args, **kwargs):
    return from_json(json.dumps(obj, *args, **kwargs))
