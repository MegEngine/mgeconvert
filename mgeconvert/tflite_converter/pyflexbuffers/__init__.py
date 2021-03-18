import json

from .fbconverter import (  # pylint: disable=import-error,no-name-in-module
    from_json,
    to_json,
)


def loads(s):
    return json.loads(to_json(bytes(s)))


def dumps(obj, *args, **kwargs):
    return from_json(json.dumps(obj, *args, **kwargs))
