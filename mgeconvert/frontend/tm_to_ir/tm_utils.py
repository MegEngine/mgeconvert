# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import inspect
from typing import Sequence

from megengine import get_logger as mge_get_logger


def get_logger(*args):
    return mge_get_logger(*args)


def _unexpand(x):
    if isinstance(x, Sequence):
        return x[0], x[1]
    elif isinstance(x, int):
        return x, x
    else:
        raise TypeError(
            "get error type! got {} expect int or tuple[int,..]".format(type(x))
        )


def _convert_kwargs_to_args(func, args, kwargs, is_bounded=False):
    # is_bounded = True when func is a method and provided args don't include 'self'
    arg_specs = inspect.getfullargspec(func)
    arg_specs_args = arg_specs.args
    if is_bounded:
        arg_specs_args = arg_specs.args[1:]
    new_args = []
    new_kwargs = {}
    new_args.extend(args)
    if set(arg_specs_args[0 : len(new_args)]) & set(kwargs.keys()):
        repeated_arg_name = set(arg_specs_args[0 : len(new_args)]) & set(kwargs.keys())
        raise TypeError(
            "{} got multiple values for argument {}".format(
                func.__qualname__, ", ".join(repeated_arg_name)
            )
        )
    if len(new_args) < len(arg_specs.args):
        for ind in range(len(new_args), len(arg_specs_args)):
            arg_name = arg_specs_args[ind]
            if arg_name in kwargs:
                new_args.append(kwargs[arg_name])
            else:
                index = ind - len(arg_specs_args) + len(arg_specs.defaults)
                assert 0 <= index < len(arg_specs.defaults)
                new_args.append(arg_specs.defaults[index])

    for kwarg_name in arg_specs.kwonlyargs:
        if kwarg_name in kwargs:
            new_kwargs[kwarg_name] = kwargs[kwarg_name]
        else:
            assert kwarg_name in arg_specs.kwonlydefaults
            new_kwargs[kwarg_name] = arg_specs.kwonlydefaults[kwarg_name]
    for k, v in kwargs.items():
        if k not in arg_specs.args and k not in arg_specs.kwonlyargs:
            if arg_specs.varkw is None:
                raise TypeError(
                    "{} got an unexpected keyword argument {}".format(
                        func.__qualname__, k
                    )
                )
            new_kwargs[k] = v
    return tuple(new_args), new_kwargs
