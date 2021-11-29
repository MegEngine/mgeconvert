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
from megengine.traced_module.expr import CallFunction, CallMethod
from megengine.utils.tuple_function import _pair, _pair_nonzero

from ....converter_ir.ir_op import AdaptiveAvgPool2dOpr, AvgPool2dOpr, MaxPool2dOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)

mode_map = {
    M.MaxPool2d: MaxPool2dOpr,
    F.max_pool2d: MaxPool2dOpr,
    M.AvgPool2d: AvgPool2dOpr,
    F.avg_pool2d: AvgPool2dOpr,
}


@_register_op(
    M.MaxPool2d,
    F.max_pool2d,
    M.AvgPool2d,
    F.avg_pool2d,
    M.adaptive_pooling.AdaptiveAvgPool2d,
    F.adaptive_avg_pool2d,
)
class GenPool2dOpr(OpGenBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph)

        if isinstance(expr, CallMethod):
            m = expr.inputs[0].owner
            self.kernel_size = _pair_nonzero(m.kernel_size)
            self.stride = _pair_nonzero(m.stride)
            self.padding = _pair(m.padding)
            op_cls = mode_map[type(m)]
        elif isinstance(expr, CallFunction):
            self.kernel_size = _pair_nonzero(self.args[1])
            self.stride = self.kernel_size
            self.padding = (0, 0)
            if self.args[2] is not None:
                self.stride = _pair_nonzero(self.args[2])
            if self.args[3] is not None:
                self.padding = _pair(self.args[3])
            op_cls = mode_map[expr.func]
        self.op = op_cls(self.kernel_size, self.stride, self.padding)
        self.add_opr_vars()

    def add_opr_vars(self):
        if isinstance(self.expr, CallMethod):
            inp = self.args[1]
        elif isinstance(self.expr, CallFunction):
            inp = self.args[0]
        inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
        self.op.add_inp_tensors(inp_tensor)
        self.add_opr_out_tensors()


@_register_op(M.adaptive_pooling.AdaptiveAvgPool2d, F.adaptive_avg_pool2d)
class GenAdaptiveAvgPool2dOpr(OpGenBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph)

        if isinstance(expr, CallMethod):
            m = expr.inputs[0].owner
            self.op = AdaptiveAvgPool2dOpr(m.oshp)
        elif isinstance(expr, CallFunction):
            self.op = AdaptiveAvgPool2dOpr(self.args[1])
        self.add_opr_vars()

    def add_opr_vars(self):
        if isinstance(self.expr, CallMethod):
            inp = self.args[1]
        elif isinstance(self.expr, CallFunction):
            inp = self.args[0]
        inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
        self.op.add_inp_tensors(inp_tensor)
        self.add_opr_out_tensors()
