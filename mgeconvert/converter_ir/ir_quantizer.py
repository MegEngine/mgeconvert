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
        if tensor.q_dtype is None:
            return value

        if tensor.scale:
            value = value / tensor.scale
            value = np.round(value)
        if tensor.zero_point:
            value += tensor.zero_point
        np_dtype = tensor.np_dtype
        dt = np.dtype(np_dtype)
        if tensor.qmin is not None and tensor.qmax is not None:
            v_min = tensor.qmin
            v_max = tensor.qmax
        else:
            assert np.issubdtype(dt, np.integer)
            v_min = np.iinfo(dt).min
            v_max = np.iinfo(dt).max
        value = np.clip(value, v_min, v_max)
        value = value.astype(np_dtype)
        return value

    def save_quantize_params(self, irgraph):
        all_tensors = set()
        for opr in irgraph.all_oprs:
            for t in opr.inp_tensors + opr.out_tensors:
                all_tensors.add(t)
        for t in all_tensors:
            self.parse_quant_info(t)

    def fake_quant(self, t: IRTensor):
        assert t.q_dtype is not None and t.np_data is not None
        inp = megengine.tensor(t.np_data, dtype="float32")
        scale = megengine.tensor([float(t.scale)])
        zp = float(t.zero_point) if t.zero_point else 0.0
        zero_point = megengine.tensor([zp])
        if t.qmin is not None and t.qmax is not None:
            v_min = t.qmin
            v_max = t.qmax
        else:
            dt = np.dtype(t.np_dtype)
            assert np.issubdtype(dt, np.integer)
            v_min = np.iinfo(dt).min
            v_max = np.iinfo(dt).max
        from megengine.core._imperative_rt.core2 import (  # pylint:disable=import-error
            apply,
        )
        from megengine.core.ops.builtin import FakeQuant

        return apply(FakeQuant(qmin=v_min, qmax=v_max), inp, scale, zero_point)[
            0
        ].numpy()

    def get_quant_info(self, t: IRTensor):
        assert t.q_dtype is not None
        assert isinstance(t.q_dtype, str)
        np_dtype = t.np_dtype
        q_dtype = t.q_dtype[1:] if t.q_dtype[0] == "q" else t.q_dtype
        try:
            dt = np.dtype(np_dtype)
        except TypeError:
            dt = None

        v_max, v_min = None, None
        is_weight = t.np_data is not None
        if t.qmin is not None and t.qmax is not None:
            v_min = t.qmin
            v_max = t.qmax
        elif dt is not None and np.issubdtype(dt, np.integer):
            v_min = np.iinfo(dt).min
            v_max = np.iinfo(dt).max
        assert v_max is not None and v_min is not None
        return {
            "dtype": q_dtype,
            "qmin": v_min,
            "qmax": v_max,
            "scale": t.scale,
            "zero_point": t.zero_point,
            "is_weight": is_weight,
        }

    def set_quant_info(self, name, t: IRTensor):
        """Set tensor named `name` the same quant info as tensor `t`.
        """
        self.quant_params[name] = self.get_quant_info(t)

    def parse_quant_info(self, t: IRTensor):
        if t.q_dtype is None:
            return
        is_weight = t.np_data is not None
        if self.param_fake_quant and is_weight:
            t.np_data = self.fake_quant(t)
        else:
            self.quant_params[t.name] = self.get_quant_info(t)

    def dump_quant_param(self, path="quant_params.json"):
        if len(self.quant_params) == 0:
            return
        params = json.dumps(self.quant_params, indent=4)
        with open(path, "w") as f:
            f.write(params)
