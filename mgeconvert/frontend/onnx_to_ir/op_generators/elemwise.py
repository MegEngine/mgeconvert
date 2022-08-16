# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ....converter_ir.ir_op import (
    AddOpr,
    DivOpr,
    HardSigmoidOpr,
    MatMulOpr,
    MulOpr,
    PowOpr,
    ReduceOpr,
    ReluOpr,
    SigmoidOpr,
    SoftmaxOpr,
    SqrtOpr,
    SubOpr,
)
from .base import OpGenBase, _register_op


@_register_op("Mul")
class GenMulOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = MulOpr()
        self.add_tensors()


@_register_op("Add")
class GenAddOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = AddOpr()
        self.add_tensors()


@_register_op("Sub")
class GenSubOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = SubOpr()
        self.add_tensors()


@_register_op("Pow")
class GenPowOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = PowOpr()
        self.add_tensors()


@_register_op("Sqrt")
class GenSqrtOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = SqrtOpr()
        self.add_tensors()


@_register_op("Div")
class GenDivOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = DivOpr()
        self.add_tensors()


@_register_op("MatMul")
class GenMatMulOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = MatMulOpr()
        self.add_tensors()


@_register_op("ReduceMean")
class GenReduceMeanOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)
        self.mode = "MEAN"
        self.keepdims = True
        for attr in node.attribute:
            if attr.name == "axes":
                self.axis = tuple(attr.ints)
            elif attr.name == "keepdims":
                self.keepdims = attr.i == 1
        self.op = ReduceOpr(self.axis, self.mode, self.keepdims)
        self.add_tensors()


@_register_op("Sigmoid")
class GenSigmoidOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = SigmoidOpr()
        self.add_tensors()


@_register_op("HardSigmoid")
class GenHardSigmoidOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = HardSigmoidOpr()
        self.add_tensors()


@_register_op("Relu")
class GenReluOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = ReluOpr()
        self.add_tensors()


@_register_op("Softmax")
class GenSoftmaxOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        assert len(node.attribute) == 1, "Softmax's attribute should be only 1"
        self.axis = node.attribute[0].i
        self.op = SoftmaxOpr(self.axis)
        self.add_tensors()
