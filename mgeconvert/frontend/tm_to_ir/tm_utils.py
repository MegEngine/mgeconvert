# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import inspect
from typing import Sequence, Union

import numpy as np
from megengine import Tensor
from megengine import get_logger as mge_get_logger
from megengine.core.tensor import dtype
from megengine.core.tensor.dtype import QuantDtypeMeta
from megengine.quantization.utils import create_qparams


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


def _update_inputs_qparams(
    traced_module,
    input_data_type: Union[str, QuantDtypeMeta],
    input_scales,
    input_zero_points,
):
    if input_data_type is None or input_scales is None:
        return
    for i in range(len(traced_module.graph.inputs[1:])):
        if traced_module.graph.inputs[i + 1].qparams is None:
            traced_module.graph.inputs[i + 1].qparams = create_qparams()
        if input_data_type in dtype._builtin_quant_dtypes:
            q_dtype_meta = dtype._builtin_quant_dtypes[input_data_type]
        elif isinstance(input_data_type, dtype.QuantDtypeMeta):
            q_dtype_meta = input_data_type
        else:
            assert isinstance(input_data_type, str)
            dt = np.dtype(input_data_type)
            assert np.issubdtype(dt, np.integer)
            v_min = np.iinfo(dt).min
            v_max = np.iinfo(dt).max
            q_dtype_meta = dtype.QuantDtypeMeta(
                input_data_type, "", input_data_type, v_min, v_max
            )
        traced_module.graph.inputs[i + 1].qparams.dtype_meta = q_dtype_meta
    if input_scales is not None:
        if isinstance(input_scales, str):
            str_scales = input_scales.split(",")
            input_scales = []
            try:
                for s in str_scales:
                    input_scales.append(float(s))
            except:
                raise ValueError(
                    "input scales({}) do not in correct format.".format(str_scales)
                )
        if not isinstance(input_scales, Sequence):
            input_scales = (input_scales,)
        for i in range(len(traced_module.graph.inputs[1:])):
            scale = input_scales[i] if i < len(input_scales) else input_scales[-1]
            traced_module.graph.inputs[i + 1].qparams.scale = Tensor(float(scale))
    if input_zero_points is not None:
        if isinstance(input_zero_points, str):
            str_zp = input_zero_points.split(",")
            input_zero_points = []
            try:
                for zp in str_zp:
                    input_zero_points.append(float(zp))
            except:
                raise ValueError(
                    "input zero points({}) do not in correct format.".format(str_zp)
                )
        if not isinstance(input_zero_points, Sequence):
            input_zero_points = (input_zero_points,)
        for i in range(len(traced_module.graph.inputs[1:])):
            zero_point = (
                input_zero_points[i]
                if i < len(input_zero_points)
                else input_zero_points[-1]
            )
            traced_module.graph.inputs[i + 1].qparams.zero_point = Tensor(
                int(zero_point)
            )
