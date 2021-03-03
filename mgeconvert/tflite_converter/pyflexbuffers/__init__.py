import json

from .fbconverter import from_json, to_json  # pylint: disable=no-name-in-module


def loads(s):
    return json.loads(to_json(bytes(s)))


def dumps(obj, *args, **kwargs):
    return from_json(json.dumps(obj, *args, **kwargs))
