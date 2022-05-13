# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module

import megengine.functional as F
import megengine.module as M
import numpy as np
from megengine.traced_module.expr import CallFunction, CallMethod
from mgeconvert.converter_ir.ir_tensor import IRTensor

from ....converter_ir.ir_op import SoftmaxOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


def get_softmax_axis(ndim: int) -> int:
    if ndim in (0, 1, 3):
        return 0
    return 1


@_register_op(M.Softmax, F.softmax)
class GenSoftmaxOpr(OpGenBase):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph)
        if isinstance(self.expr, CallMethod):
            module = expr.inputs[0].owner
            self.axis = module.axis
        elif isinstance(self.expr, CallFunction):
            self.axis = self.args[1]

        self.op = SoftmaxOpr(self.axis)
        self.add_opr_vars()
        assert self.op.inp_tensors[0].ndim in [
            1,
            2,
            4,
        ], "Softmax do not support {} dim".format(self.op.inp_tensors[0].ndim)
        if self.op.axis is None:
            self.op.axis = get_softmax_axis(self.op.inp_tensors[0].ndim)

    def add_opr_vars(self):
        if isinstance(self.expr, CallMethod):
            inp_tensor = self.resolver.get_ir_tensor(self.args[1], user_opr=self.op)
            self.op.add_inp_tensors(inp_tensor)

        elif isinstance(self.expr, CallFunction):
            inp_tensor = self.resolver.get_ir_tensor(self.args[0], user_opr=self.op)
            self.op.add_inp_tensors(inp_tensor)

        # self.add_axis()
        self.add_opr_out_tensors()

    def add_axis(self):
        if self.axis is None:
            self.axis = get_softmax_axis(self.op.inp_tensors[0].ndim)
        axis_tensor = IRTensor(
            self.op.inp_tensors[0].name + "_softmax_axis",
            (1,),
            np.int32,
            np_data=np.array(self.axis),
            axis=None,
        )
        self.op.add_inp_tensors(axis_tensor)
