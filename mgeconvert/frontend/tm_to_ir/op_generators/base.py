# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import ABC

from megengine.module import Module
from megengine.tensor import Tensor
from megengine.traced_module.expr import CallFunction, CallMethod
from megengine.traced_module.node import ModuleNode, TensorNode
from mgeconvert.converter_ir.ir_op import OpBase

from ..tm_tensor_resolver import TensorNodeResolver
from ..tm_utils import _convert_kwargs_to_args

EXPR2OP = {}


def _register_op(*ops):
    def callback(impl):
        for op in ops:
            EXPR2OP[op] = impl
        return impl

    return callback


class OpGenBase(ABC):
    def __init__(self, expr, irgraph) -> None:
        self.expr = expr
        self.irgraph = irgraph
        self.resolver = TensorNodeResolver(self.irgraph)
        self.op = OpBase()
        if isinstance(self.expr, CallFunction):
            args, kwargs = _convert_kwargs_to_args(expr.func, expr.args, expr.kwargs)
            self.args = args
            self.kwargs = kwargs
        if isinstance(self.expr, CallMethod):
            if isinstance(expr.args[0], type):
                obj_type = expr.args[0]
            elif isinstance(expr.args[0], ModuleNode):
                obj_type = expr.args[0].module_type
            else:
                assert isinstance(expr.args[0], TensorNode)
                obj_type = Tensor
            meth = getattr(
                obj_type, "forward" if issubclass(obj_type, Module) else expr.method
            )
            args, kwargs = _convert_kwargs_to_args(meth, expr.args, expr.kwargs)
            self.args = args
            self.kwargs = kwargs

    def get_opr(self):
        return self.op

    def add_opr_out_tensors(self):
        if len(self.op.inp_tensors) > 0:
            is_qat = (
                hasattr(self.op.inp_tensors[0], "scale")
                and self.op.inp_tensors[0].scale is not None
            )
        else:
            is_qat = False
        for o in self.expr.outputs:
            t = self.resolver.get_ir_tensor(o, owner_opr=self.op)
            if is_qat:
                t.set_qparams_from_other_tensor(self.op.inp_tensors[0])
            self.op.add_out_tensors(t)
