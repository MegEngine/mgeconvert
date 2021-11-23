# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module,no-member

import megengine.functional as F
from megengine.traced_module.expr import CallFunction

from ....converter_ir.ir_op import ResizeOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


@_register_op(F.vision.interpolate)
class GenResizeOpr(OpGenBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph)
        assert isinstance(expr, CallFunction)
        self.size = expr.kwargs["size"] if "size" in expr.kwargs.keys() else None
        self.scale_factor = (
            expr.kwargs["scale_factor"] if "scale_factor" in expr.kwargs else None
        )
        self.mode = expr.kwargs["mode"] if "mode" in expr.kwargs else None
        self.align_corners = (
            expr.kwargs["align_corners"] if "align_corners" in expr.kwargs else False
        )
        self.op = ResizeOpr(self.size, self.scale_factor, self.mode, self.align_corners)
        self.add_opr_vars()

    def add_opr_vars(self):
        inp = self.expr.args[0]
        inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
        self.op.add_inp_tensors(inp_tensor)
        self.add_opr_out_tensors()
