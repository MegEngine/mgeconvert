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
import megengine.module.qat as QAT
from megengine.traced_module.expr import CallFunction, CallMethod

from ....converter_ir.ir_op import LinearOpr, MatMulOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


@_register_op(F.matmul)
class GenMatMulOpr(OpGenBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph)
        if isinstance(expr, CallFunction):
            self.transpose_a = self.args[2]
            self.transpose_b = self.args[3]
            self.compute_mode = self.args[4]
            self.format = self.args[5]
            self.op = MatMulOpr(
                self.transpose_a, self.transpose_b, self.compute_mode, self.format
            )
        self.add_opr_vars()

    def add_opr_vars(self):
        for inp in self.expr.inputs:
            inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
            self.op.add_inp_tensors(inp_tensor)
        self.add_opr_out_tensors()


@_register_op(F.linear, M.Linear)
class GenLinearOpr(OpGenBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph)
        if isinstance(expr, CallMethod):
            m = expr.inputs[0].owner
            self.weight = m.weight
            self.has_bias = bool(m.bias is not None)
        elif isinstance(expr, CallFunction):
            self.has_bias = bool(len(expr.inputs) == 3)
        self.op = LinearOpr(self.has_bias)

        self.add_opr_vars()

    def add_opr_vars(self):
        if isinstance(self.expr, CallMethod):
            for inp in self.expr.inputs[1:]:
                inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
                self.op.add_inp_tensors(inp_tensor)
            self.add_const_inputs()
        else:
            for inp in self.expr.inputs:
                inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
                self.op.add_inp_tensors(inp_tensor)
        self.add_opr_out_tensors()

    def add_const_inputs(self):
        weight_tensor = self.resolver.get_ir_tensor(
            self.weight, user_opr=self.op, name=self.expr.inputs[0]._name + "_weight",
        )
        self.op.add_inp_tensors(weight_tensor)
        if self.has_bias:
            bias_tensor = self.resolver.get_ir_tensor(
                self.expr.inputs[0].owner.bias,
                user_opr=self.op,
                name=self.expr.inputs[0]._name + "_bias",
            )
            self.op.add_inp_tensors(bias_tensor)


@_register_op(QAT.Linear)
class GenQLinearOpr(GenLinearOpr):
    def __init__(self, expr, irgraph):
        self.module = expr.inputs[0].owner
        if hasattr(self.module.weight_fake_quant, "get_qparams"):
            self.weight_qparams = self.module.weight_fake_quant.get_qparams()
            self.weight_dtype = self.weight_qparams.dtype_meta.np_dtype_str
            self.act_qparams = self.module.act_fake_quant.get_qparams()
            self.act_dtype = self.act_qparams.dtype_meta.np_dtype_str
        elif hasattr(self.module.weight_observer, "get_qparams"):
            self.weight_qparams = self.module.weight_observer.get_qparams()
            self.weight_dtype = self.weight_qparams.dtype_meta.np_dtype_str
            self.act_qparams = self.module.act_observer.get_qparams()
            self.act_dtype = self.act_qparams.dtype_meta.np_dtype_str
        else:
            logger.error("Observer and FakeQuantize do not have get_qparams().")
        super().__init__(expr, irgraph)

    def add_const_inputs(self):
        weight_tensor = self.resolver.get_ir_tensor(
            self.weight, user_opr=self.op, name=self.expr.inputs[0]._name + "_weight",
        )
        weight_tensor.set_qparams(
            *self.resolver.resolve_qparams(
                self.weight_qparams.scale, self.weight_qparams.zero_point
            )
        )
        weight_tensor.q_dtype = self.weight_dtype
        self.op.add_inp_tensors(weight_tensor)
        if self.has_bias:
            bias_tensor = self.resolver.get_ir_tensor(
                self.expr.inputs[0].owner.bias,
                user_opr=self.op,
                name=self.expr.inputs[0]._name + "_bias",
            )
            bias_tensor.set_qparams(
                self.op.inp_tensors[0].scale * weight_tensor.scale, 0
            )
            bias_tensor.q_dtype = "int32"
            self.op.add_inp_tensors(bias_tensor)

    def add_opr_out_tensors(self):
        for o in self.expr.outputs:
            out_tensor = self.resolver.get_ir_tensor(o, owner_opr=self.op)
            out_tensor.set_qparams(
                *self.resolver.resolve_qparams(
                    self.act_qparams.scale, self.act_qparams.zero_point
                )
            )
            out_tensor.q_dtype = self.act_dtype
            self.op.add_out_tensors(out_tensor)
