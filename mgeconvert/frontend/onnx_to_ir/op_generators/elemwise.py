# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ....converter_ir.ir_op import AddOpr, MulOpr, ReluOpr, SigmoidOpr, SoftmaxOpr
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


@_register_op("Sigmoid")
class GenSigmoidOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = SigmoidOpr()
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
