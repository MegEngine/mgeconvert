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

from ....converter_ir.ir_op import DropoutOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


@_register_op(M.Dropout, F.dropout)
class GenDropoutOpr(OpGenBase):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph)
        if isinstance(expr, CallMethod):
            if hasattr(expr.args[0].owner, "drop_prob"):
                self.drop_prob = expr.args[0].owner.drop_prob
            else:
                self.drop_prob = 0
            self.training = False

        if isinstance(expr, CallFunction):
            assert False, "functional.dropout is not implement"

        self.op = DropoutOpr(self.drop_prob, self.training)
        self.add_opr_vars()

    def add_opr_vars(self):
        if isinstance(self.expr, CallMethod):
            for i in self.args[1:]:
                t = self.resolver.get_ir_tensor(i, user_opr=self.op)
                self.op.add_inp_tensors(t)
        self.add_opr_out_tensors()
