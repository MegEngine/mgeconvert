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

from ....converter_ir.ir_op import BatchNormalizationOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


@_register_op(M.BatchNorm1d, M.BatchNorm2d, F.batch_norm)
class GenBatchNormalizationOpr(OpGenBase):
    def __init__(self, expr, ir_graph):
        super().__init__(expr, ir_graph)
        if isinstance(self.expr, CallMethod):
            bn_module = expr.inputs[0].owner
            state_dict = bn_module.state_dict()
            self.running_mean = state_dict["running_mean"].squeeze()
            self.running_var = state_dict["running_var"].squeeze()
            self.weight = state_dict["weight"].squeeze()
            self.bias = state_dict["bias"].squeeze()
        elif isinstance(self.expr, CallFunction):
            assert False, "not inplement function batchnorm"
        self.op = BatchNormalizationOpr(eps=bn_module.eps, momentum=bn_module.momentum,)
        self.add_opr_vars()

    def add_opr_vars(self):
        if isinstance(self.expr, CallMethod):
            for i in self.args[1:]:
                t = self.resolver.get_ir_tensor(i, user_opr=self.op)
                self.op.add_inp_tensors(t)
            self.add_const_inputs()
        elif isinstance(self.expr, CallFunction):
            assert False, "not inplement function batchnorm"

        self.add_opr_out_tensors()

    def add_opr_out_tensors(self):
        for o in self.expr.outputs:
            out_tensor = self.resolver.get_ir_tensor(o, self.op)
            self.op.add_out_tensors(out_tensor)

    def add_const_inputs(self):
        if self.weight is not None:
            weight_tensor = self.resolver.get_ir_tensor(
                self.weight,
                owner_opr=None,
                name=self.expr.inputs[0]._name + "_weight",
                user_opr=self.op,
            )
            self.op.add_inp_tensors(weight_tensor)
        if self.bias is not None:
            bias_tensor = self.resolver.get_ir_tensor(
                self.bias,
                owner_opr=None,
                name=self.expr.inputs[0]._name + "_bias",
                user_opr=self.op,
            )
            self.op.add_inp_tensors(bias_tensor)

        if self.running_mean is not None:
            mean_tensor = self.resolver.get_ir_tensor(
                self.running_mean,
                owner_opr=None,
                name=self.expr.inputs[0]._name + "_runing_mean",
                user_opr=self.op,
            )
            self.op.add_inp_tensors(mean_tensor)

        if self.running_var is not None:
            var_tensor = self.resolver.get_ir_tensor(
                self.running_var,
                owner_opr=None,
                name=self.expr.inputs[0]._name + "_runing_var",
                user_opr=self.op,
            )
            self.op.add_inp_tensors(var_tensor)
