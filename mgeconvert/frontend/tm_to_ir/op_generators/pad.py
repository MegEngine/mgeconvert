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
from megengine.core._imperative_rt import ops
from megengine.traced_module.expr import Apply, CallFunction, CallMethod

from ....converter_ir.ir_op import PadOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


@_register_op(M.Pad, F.nn.pad, "Padding")
class GenPadOpr(OpGenBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph)
        self.mode = "constant"
        self.constant_value = 0.0

        if isinstance(expr, CallMethod):
            m = expr.inputs[0].owner
            self.pad_width = m.pad_width
            self.mode = m.mode
            self.pad_val = m.pad_val
        elif isinstance(expr, CallFunction):
            self.pad_width = self.args[1]
            if len(self.args) > 2:
                self.mode = self.args[2]
            if len(self.args) > 3:
                self.pad_val = self.args[3]
        elif isinstance(expr, Apply):
            opdef = expr.opdef
            self.pad_width = (
                (opdef.front_offset_dim0, opdef.back_offset_dim0),
                (opdef.front_offset_dim1, opdef.back_offset_dim1),
                (opdef.front_offset_dim2, opdef.back_offset_dim2),
                (opdef.front_offset_dim3, opdef.back_offset_dim3),
            )
            if opdef.padding_mode == ops.Padding.PaddingMode.CONSTANT:
                self.mode = "constant"
            elif opdef.padding_mode == ops.Padding.PaddingMode.REFLECT:
                self.mode = "reflect"
            elif opdef.padding_mode == ops.Padding.PaddingMode.REPLICATE:
                self.mode = "replicate"
            self.pad_val = opdef.padding_val
        self.op = PadOpr(self.pad_width, self.mode, self.pad_val)
        self.add_opr_vars()

    def add_opr_vars(self):
        if isinstance(self.expr, CallMethod):
            inp = self.args[1]
        elif isinstance(self.expr, CallFunction):
            inp = self.args[0]
        elif isinstance(self.expr, Apply):
            inp = self.expr.inputs[0]
        inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
        self.op.add_inp_tensors(inp_tensor)
        self.add_opr_out_tensors()
