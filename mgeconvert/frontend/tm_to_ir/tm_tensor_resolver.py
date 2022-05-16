# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module

import megengine as mge
import numpy as np
from megengine.traced_module.expr import Constant, GetAttr
from megengine.traced_module.node import TensorNode

from ...converter_ir.ir_graph import IRGraph
from ...converter_ir.ir_tensor import AxisOrder, IRTensor


class TensorNodeResolver:
    __const_id = 0

    def __init__(self, irgraph: IRGraph) -> None:
        self.irgraph = irgraph

    def resolve(self, inp, owner_opr=None, param_name=None, axis_order=AxisOrder.NCHW):
        scale = None
        zero_point = None

        if isinstance(inp, TensorNode):
            shape = inp.shape
            dtype = inp.dtype
            if isinstance(inp.expr, Constant):
                np_data = inp.expr.value.numpy()
            elif isinstance(inp.expr, GetAttr):
                np_data = inp.expr.interpret(inp.expr.inputs[0].owner)[0].numpy()
            else:
                np_data = None
            name = inp._name
            ori_id = inp._id
        elif isinstance(inp, (int, float, list, np.ndarray)):
            np_data = np.array(inp)
            np_data = np_data.astype(np_data.dtype.name.replace("64", "32"))
            dtype = np_data.dtype.type
            shape = np_data.shape
            name = (
                param_name
                if param_name
                else "const_val_" + str(TensorNodeResolver.__const_id)
            )
            TensorNodeResolver.__const_id += 1
            ori_id = None
        elif isinstance(inp, mge.Tensor):
            name = param_name
            shape = inp.shape
            dtype = inp.dtype
            np_data = inp.numpy()
            ori_id = None

        return (
            IRTensor(
                name,
                shape,
                dtype,
                scale=scale,
                zero_point=zero_point,
                np_data=np_data,
                owner_opr=owner_opr,
                axis=axis_order,
            ),
            ori_id,
        )

    def get_ir_tensor(
        self, inp, owner_opr=None, user_opr=None, name=None, axis_order=AxisOrder.NCHW
    ):
        ir_tensor, ori_id = self.resolve(
            inp, owner_opr=owner_opr, param_name=name, axis_order=axis_order
        )
        ori_tensor = self.irgraph.get_tensor(ori_id, ir_tensor)
        if user_opr is not None and user_opr not in ori_tensor.user_opr:
            ori_tensor.add_user_opr(user_opr)
        return ori_tensor

    def resolve_qparams(self, scale, zero_point):
        if isinstance(scale, mge.Tensor):
            scale = scale.numpy()
        if zero_point:
            if isinstance(scale, mge.Tensor):
                zero_point = zero_point.numpy()
        return scale, zero_point
