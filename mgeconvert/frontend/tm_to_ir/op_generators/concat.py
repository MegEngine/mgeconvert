# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module

from collections import Iterable

import megengine.functional as F
import megengine.module as M
import megengine.module.qat as QAT
from megengine.traced_module.expr import CallFunction, CallMethod

from ....converter_ir.ir_op import ConcatOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


@_register_op(F.concat, M.Concat)
class GenConcatOpr(OpGenBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph)
        if isinstance(self.expr, CallMethod):
            self.axis = self.args[2]
        elif isinstance(self.expr, CallFunction):
            self.axis = self.args[1]
        self.op = ConcatOpr(self.axis)
        self.add_opr_vars()

    def add_opr_vars(self):
        if isinstance(self.expr, CallMethod):
            inp_data = self.args[1]
        elif isinstance(self.expr, CallFunction):
            inp_data = self.args[0]
        assert isinstance(inp_data, Iterable), "Concat inputs must be Iterable."
        for i in inp_data:
            t = self.resolver.get_ir_tensor(i, user_opr=self.op)
            self.op.add_inp_tensors(t)
        self.add_opr_out_tensors()


@_register_op(QAT.Concat)
class GenQConcatOpr(GenConcatOpr):
    def __init__(self, expr, irgraph) -> None:
        if isinstance(expr, CallMethod):
            self.module = expr.inputs[0].owner
        if hasattr(self.module.act_fake_quant, "get_qparams"):
            self.act_qparams = self.module.act_fake_quant.get_qparams()
            self.act_dtype = self.act_qparams.dtype_meta.name
        elif hasattr(self.module.act_observer, "get_qparams"):
            self.act_qparams = self.module.act_observer.get_qparams()
            self.act_dtype = self.act_qparams.dtype_meta.name
        else:
            logger.error("Observer and FakeQuantize do not have get_qparams().")
        super().__init__(expr, irgraph=irgraph)

    def add_opr_out_tensors(self):
        for o in self.expr.outputs:
            t = self.resolver.get_ir_tensor(o, owner_opr=self.op)
            t.set_qparams_from_mge_qparams(self.act_qparams)
            t.q_dtype = self.act_dtype
            self.op.add_out_tensors(t)
