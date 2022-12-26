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
    SqueezeOpr,
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


@_register_op("Add", "Sum")
class GenAddOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = AddOpr()
        self.add_tensors()
        assert len(self.op.inp_tensors) == 2, "Sum of more than two is not supported"


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
        if len(node.attribute) != 0:
            assert (
                len(node.attribute) == 1
            ), "Softmax's attribute should be only 1 when exist"
            self.axis = node.attribute[0].i
            self.op = SoftmaxOpr(self.axis)
            self.add_tensors()
        else:
            self.axis = 1
            self.op = SoftmaxOpr(self.axis)
            self.add_tensors()
            inp_shape = self.op.inp_tensors[0].shape
            assert (
                len(inp_shape) == 2 and inp_shape[0] == 1
            ), "Softmax's input shape dims should be only 1 x N when attribute not exist"


@_register_op("Squeeze")
class GenSqueezeOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)
        for attr in node.attribute:
            if attr.name == "axes":
                self.axis = tuple(attr.ints)
        self.op = SqueezeOpr(self.axis)
        self.add_tensors()
