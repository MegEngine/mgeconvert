# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module,no-member

from abc import ABC

import megengine.functional as F
import megengine.module as M
import megengine.module.qat as QAT
import numpy as np
from megengine.traced_module.expr import CallFunction, CallMethod
from megengine.traced_module.node import ModuleNode, TensorNode

from ....converter_ir.ir_op import (
    AbsOpr,
    AddOpr,
    CeilOpr,
    ExpOpr,
    FloorDivOpr,
    FloorOpr,
    FuseAddReluOpr,
    HardSigmoidOpr,
    HardSwishOpr,
    IdentityOpr,
    LeakyReluOpr,
    LogOpr,
    MaxOpr,
    MinOpr,
    MulOpr,
    PowOpr,
    Relu6Opr,
    ReluOpr,
    SigmoidOpr,
    SiLUOpr,
    SubOpr,
    TanHOpr,
    TrueDivOpr,
)
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


@_register_op(F.leaky_relu)
class GenLeakyReluOpr(OpGenBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph)
        assert isinstance(self.expr, CallFunction)
        self.negative_slope = self.args[1]
        self.op = LeakyReluOpr(self.negative_slope)
        self.add_opr_vars()

    def add_opr_vars(self):
        inp = self.args[0]
        inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
        self.op.add_inp_tensors(inp_tensor)
        self.add_opr_out_tensors()


class GenElemwiseOpr(OpGenBase, ABC):
    def __init__(self, expr, irgraph, op_cls):
        super().__init__(expr, irgraph)
        self.op = op_cls()
        self.add_opr_vars()

    def add_opr_vars(self):
        for inp in self.args:
            if isinstance(inp, ModuleNode):
                continue
            assert isinstance(
                inp, (TensorNode, int, float, np.ndarray)
            ), "expr inputs type not support {}".format(type(inp))
            tm_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
            if isinstance(inp, (int, float, np.ndarray)):
                target_dtype = self.op.inp_tensors[0].dtype
                tm_tensor.set_dtype(target_dtype)
            self.op.add_inp_tensors(tm_tensor)

        for oup in self.expr.outputs:
            assert isinstance(
                oup, (TensorNode)
            ), "expr outputs type not support {}".format(type(oup))
            tm_tensor = self.resolver.get_ir_tensor(oup, owner_opr=self.op)
            self.op.add_out_tensors(tm_tensor)
        if (
            hasattr(self.op.inp_tensors[0], "scale")
            and self.op.inp_tensors[0].scale is not None
        ):
            for o in self.op.out_tensors:
                o.set_qparams_from_other_tensor(self.op.inp_tensors[0])
            # set dtype for const value


@_register_op("__add__", "__iadd__")
class GenAddOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, AddOpr)


@_register_op("__sub__")
class GenSubOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, SubOpr)


@_register_op("__mul__")
class GenMulOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, MulOpr)


@_register_op("__truediv__")
class GenTrueDivOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, TrueDivOpr)


@_register_op("__floordiv__")
class GenFloorDivOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, FloorDivOpr)


@_register_op("__pow__")
class GenPowOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, PowOpr)


@_register_op(F.maximum)
class GenMaxOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, MaxOpr)


@_register_op(F.minimum)
class GenMinOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, MinOpr)


@_register_op(F.exp)
class GenExpOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, ExpOpr)


@_register_op(F.floor)
class GenFloorOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, FloorOpr)


@_register_op(F.ceil)
class GenCeilOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, CeilOpr)


@_register_op(F.abs)
class GenAbsOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, AbsOpr)


@_register_op(F.relu, M.activation.ReLU)
class GenReluOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, ReluOpr)


@_register_op(F.tanh)
class GenTanHOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, TanHOpr)


@_register_op(F.sigmoid, M.Sigmoid)
class GenSigmoidOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, SigmoidOpr)


@_register_op(F.hsigmoid)
class GenHardSigmoidOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, HardSigmoidOpr)
        self.add_const_vars()

    def add_const_vars(self):
        const_tensor3 = self.resolver.get_ir_tensor(3, user_opr=self.op)
        const_tensor6 = self.resolver.get_ir_tensor(6, user_opr=self.op)
        self.op.add_inp_tensors(const_tensor3)
        self.op.add_inp_tensors(const_tensor6)


@_register_op(F.log)
class GenLogOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, LogOpr)


@_register_op(F.relu6)
class GenRelu6Opr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, Relu6Opr)


@_register_op(F.silu, M.SiLU)
class GenSiLUOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, SiLUOpr)


@_register_op(F.hswish)
class GenHardSwishOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, HardSwishOpr)
        self.add_const_vars()

    def add_const_vars(self):
        const_tensor3 = self.resolver.get_ir_tensor(3, user_opr=self.op)
        const_tensor6 = self.resolver.get_ir_tensor(6, user_opr=self.op)
        self.op.add_inp_tensors(const_tensor3)
        self.op.add_inp_tensors(const_tensor6)


@_register_op(M.Identity)
class GenIdentityOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, IdentityOpr)


class GenFuseAddReluOpr(GenElemwiseOpr):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph, FuseAddReluOpr)


method_opr_map = {
    "add": GenAddOpr,
    "sigmoid": GenSigmoidOpr,
    "mul": GenMulOpr,
    "abs": GenAbsOpr,
    "ceil": GenCeilOpr,
    "exp": GenExpOpr,
    "floor": GenFloorOpr,
    "log": GenLogOpr,
    "max": GenMaxOpr,
    "min": GenMinOpr,
    "pow": GenPowOpr,
    "relu": GenReluOpr,
    "sub": GenSubOpr,
    "tanh": GenTanHOpr,
    "true_div": GenTrueDivOpr,
    "floor_div": GenFloorDivOpr,
    "relu6": GenRelu6Opr,
    "identity": GenIdentityOpr,
    "hswish": GenHardSwishOpr,
    "hsigmoid": GenHardSigmoidOpr,
    "fuse_add_relu": GenFuseAddReluOpr,
}


@_register_op(QAT.Elemwise, M.Elemwise)
def get_elemwise_op(expr, net):
    assert isinstance(expr, CallMethod)
    module = expr.inputs[0].owner
    method = module.method.lower()
    op_gen = method_opr_map[method](expr, net)

    if isinstance(module, QAT.QATModule):
        if hasattr(module.act_fake_quant, "get_qparams"):
            qparams = module.act_fake_quant.get_qparams()
        else:
            qparams = module.act_observer.get_qparams()
        for o in op_gen.get_opr().out_tensors:
            o.set_qparams_from_mge_qparams(qparams)
            o.scale = float(qparams.scale) if method != "sigmoid" else 1 / 256.0
    return op_gen
