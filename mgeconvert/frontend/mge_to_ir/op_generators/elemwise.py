# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ....converter_ir.ir_op import (
    AbsOpr,
    AddOpr,
    CeilOpr,
    ExpOpr,
    FloorOpr,
    FuseMulAdd3Opr,
    LogOpr,
    MaxOpr,
    MinOpr,
    MulOpr,
    PowOpr,
    ReluOpr,
    SigmoidOpr,
    SoftmaxOpr,
    SubOpr,
    TanHOpr,
    TrueDivOpr,
)
from .base import OpGenBase, _register_op


class GenFuseMulAdd3Oprs(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.op = FuseMulAdd3Opr()
        self.add_tensors(mge_opr)


mode_opr_map = {
    "add": AddOpr,
    "fuse_add_relu": AddOpr,
    "sigmoid": SigmoidOpr,
    "mul": MulOpr,
    "abs": AbsOpr,
    "ceil": CeilOpr,
    "exp": ExpOpr,
    "floor": FloorOpr,
    "log": LogOpr,
    "max": MaxOpr,
    "min": MinOpr,
    "pow": PowOpr,
    "relu": ReluOpr,
    "sub": SubOpr,
    "tanh": TanHOpr,
    "true_div": TrueDivOpr,
    "fuse_mul_add3": GenFuseMulAdd3Oprs,
}


@_register_op("Elemwise")
class GenElemwiseOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        try:
            self.mode = self.params["mode"]
        except RuntimeError:
            self.mode = "NONE"
        if self.mode.lower() in ["fuse_mul_add3"]:
            self.op = mode_opr_map[self.mode.lower()](mge_opr, irgraph).get_opr()
        else:
            self.op = mode_opr_map[self.mode.lower()]()
            if "RELU" in self.mode:
                self.op.activation = "RELU"
            self.add_tensors(mge_opr)


@_register_op("SoftmaxForward")
class GenSoftmaxForwardOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.op = SoftmaxOpr()
        self.add_tensors(mge_opr)
