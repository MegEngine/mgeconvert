# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint:disable=import-outside-toplevel, no-name-in-module
import json

import megengine
import numpy as np

from .ir_tensor import IRTensor


class IRQuantizer:
    def __init__(self, require_quantize=False, param_fake_quant=False):
        super().__init__()
        self.require_quantize = require_quantize
        self.param_fake_quant = param_fake_quant
        self.quant_params = {}

    def quantize(self, tensor: IRTensor):
        assert self.require_quantize, "This net do not require true quantize."
        value = tensor.np_data
        if isinstance(value, megengine.tensor):
            value = value.numpy()
        if tensor.scale:
            value = value / tensor.scale
            value = np.round(value)
        if tensor.zero_point:
            value += tensor.zero_point
        dt = (
            np.dtype(tensor.q_dtype)
            if isinstance(tensor.q_dtype, str)
            else tensor.q_dtype
        )
        if np.issubdtype(dt, np.integer):
            v_min = np.iinfo(dt).min
            v_max = np.iinfo(dt).max
            value = np.clip(value, v_min, v_max)
        value = value.astype(tensor.q_dtype)
        return value

    def save_quantize_params(self, irgraph):
        all_tensors = set()
        for opr in irgraph.all_oprs:
            for t in opr.inp_tensors + opr.out_tensors:
                all_tensors.add(t)
        for t in all_tensors:
            self.parse_quant_info(t)

    def parse_quant_info(self, t: IRTensor):
        dt = np.dtype(t.q_dtype)
        v_max, v_min = None, None
        is_weight = bool(t.np_data is not None)
        if np.issubdtype(dt, np.integer):
            v_min = np.iinfo(dt).min
            v_max = np.iinfo(dt).max
        if self.param_fake_quant and is_weight:
            if t.scale is not None:
                inp = megengine.tensor(t.np_data)
                scale = megengine.tensor(t.scale)
                zp = float(t.zero_point) if t.zero_point else 0.0
                zero_point = megengine.tensor(zp)
                from megengine.core._imperative_rt.core2 import (  # pylint:disable=import-error
                    apply,
                )
                from megengine.core.ops.builtin import FakeQuant

                t.np_data = apply(
                    FakeQuant(qmin=v_min, qmax=v_max), inp, scale, zero_point
                )[0].numpy()
        else:
            param = {
                "dtype": str(dt),
                "qmin": str(v_min),
                "qmax": str(v_max),
                "scale": str(t.scale),
                "zero_point": str(t.zero_point),
                "is_weight": is_weight,
            }
            self.quant_params[t.name] = param

    def dump_quant_param(self, path="quant_params.json"):
        params = json.dumps(self.quant_params, indent=4)
        with open(path, "w") as f:
            f.write(params)
