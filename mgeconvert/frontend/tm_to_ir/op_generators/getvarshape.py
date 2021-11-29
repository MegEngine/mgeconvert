# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ....converter_ir.ir_op import GetVarShapeOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


@_register_op("GetVarShape")
class GenGetVarShapeOpr(OpGenBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph)
        self.op = GetVarShapeOpr()
        self.add_opr_vars()

    def add_opr_vars(self):
        inp = self.expr.inputs[0]
        inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
        self.op.add_inp_tensors(inp_tensor)
        self.add_opr_out_tensors()
