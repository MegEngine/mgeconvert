# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module

import megengine.functional as F
from megengine.traced_module.expr import CallFunction, CallMethod

from ....converter_ir.ir_op import TransposeOpr
from .base import OpGenBase, _register_op


@_register_op(F.transpose, "transpose")
class GenTransposeOpr(OpGenBase):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph)
        if isinstance(self.expr, CallFunction):
            self.pattern = self.args[1]
        elif isinstance(self.expr, CallMethod):
            self.pattern = tuple(self.args[1:])
        self.op = TransposeOpr(self.pattern)
        self.add_opr_vars()

    def add_opr_vars(self):
        inp_tensor = self.resolver.get_ir_tensor(self.args[0], user_opr=self.op)
        self.op.add_inp_tensors(inp_tensor)

        self.add_opr_out_tensors()
