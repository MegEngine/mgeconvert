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
from megengine.traced_module.expr import CallMethod, CallFunction

from ....converter_ir.ir_op import CustomOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


class GenCustomOpr(OpGenBase):
    fn_set = {}
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph)
        op_name = ""
        if isinstance(self.expr, CallFunction):
            op_name = expr.func.__name__.title()
            
        if isinstance(self.expr, CallMethod):
            op_name = expr.args[0].name
        symbolic_fn, domain = GenCustomOpr.fn_set[op_name]
        symbolic_fn(self, *self.args)

        self.add_opr_vars()

    def add_opr_vars(self):
        self.add_opr_out_tensors()

    def set_op(self, op_name, domain, *args, **kwargs):
        self.op = CustomOpr(op_name, domain, kwargs)
        for i in args:
            t = self.resolver.get_ir_tensor(i, user_opr=self.op)
            self.op.add_inp_tensors(t)
        return self.op

    @staticmethod
    def _register_op(op_name, domain, symbolic_fn):
        GenCustomOpr.fn_set[op_name] = (symbolic_fn, domain)

import re
def get_ns_op_name_from_custom_op(symbolic_name):
    if not bool(
        re.match(r"^[a-zA-Z0-9-_]*::[a-zA-Z-_]+[a-zA-Z0-9-_]*$", symbolic_name)
    ):
        return "", symbolic_name

    ns, op_name = symbolic_name.split("::")
    if ns == "onnx":
        raise ValueError(
            f"Failed to register operator {symbolic_name}. {ns} domain cannot be modified."
        )

    if ns == "aten":
        ns = ""

    return ns, op_name


def register_op(op, symbolic_name, symbolic_fn):
    ns, op_name = get_ns_op_name_from_custom_op(symbolic_name)
    GenCustomOpr._register_op(op_name, ns, symbolic_fn)
    _register_op(op)(GenCustomOpr)

