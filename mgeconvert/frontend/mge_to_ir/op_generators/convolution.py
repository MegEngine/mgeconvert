# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ....converter_ir.ir_op import (
    AvgPool2dOpr,
    BatchNormalizationOpr,
    Conv2dOpr,
    ConvolutionBackwardFilterOpr,
    Deconv2dOpr,
    MaxPool2dOpr,
)
from ....converter_ir.ir_tensor import AxisOrder
from ..mge_utils import get_shape
from .base import OpGenBase, _register_op


@_register_op("ConvolutionForward")
class GenConv2dOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.kernel_shape = get_shape(mge_opr.inputs[1])

        self.stride = [self.params["stride_h"], self.params["stride_w"]]
        self.padding = [self.params["pad_h"], self.params["pad_w"]]
        self.dilation = [self.params["dilate_h"], self.params["dilate_w"]]
        self.groups = self.kernel_shape[0] if len(self.kernel_shape) == 5 else 1

        self.data_format = self.params["format"]
        assert self.data_format == "NCHW", "do not support this {} format".format(
            self.data_format
        )
        assert self.params["compute_mode"].lower() == "default"
        assert self.params["mode"].lower() == "cross_correlation"

        self.op = Conv2dOpr(self.stride, self.padding, self.dilation, self.groups)
        self.add_tensors(mge_opr)

    def add_tensors(self, mge_opr):
        self.op.add_inp_tensors(
            self.resolver.get_ir_tensor(mge_opr.inputs[0], user_opr=self.op)
        )
        weight_tensor = self.resolver.get_ir_tensor(
            mge_opr.inputs[1], user_opr=self.op, axis_order=AxisOrder.OIHW
        )
        weight_tensor.axis_order = AxisOrder.OIHW
        self.op.add_inp_tensors(weight_tensor)
        if len(mge_opr.inputs) > 2:
            self.op.add_inp_tensors(
                self.resolver.get_ir_tensor(mge_opr.inputs[2], user_opr=self.op)
            )

        for x in mge_opr.outputs:
            self.op.add_out_tensors(self.resolver.get_ir_tensor(x, owner_opr=self.op))


@_register_op("ConvBiasForward")
class GenConvBiasForwardOpr(GenConv2dOpr):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.op.activation = self.params["nonlineMode"]


@_register_op("ConvolutionBackwardData")
class GenDeconv2dOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph) -> None:
        super().__init__(mge_opr, irgraph)
        self.kernel_shape = get_shape(mge_opr.inputs[0])

        self.stride = [self.params["stride_h"], self.params["stride_w"]]
        self.padding = [self.params["pad_h"], self.params["pad_w"]]
        self.dilation = [self.params["dilate_h"], self.params["dilate_w"]]
        self.groups = self.kernel_shape[0] if len(self.kernel_shape) == 5 else 1

        self.sparse = self.params["sparse"]
        self.data_format = self.params["format"]
        assert self.data_format == "NCHW", "do not support this {} format".format(
            self.data_format
        )
        assert self.params["compute_mode"].lower() == "default"
        assert self.params["mode"].lower() == "cross_correlation"

        self.op = Deconv2dOpr(self.stride, self.padding, self.dilation, self.groups)
        self.add_tensors(mge_opr)

    def add_tensors(self, mge_opr):
        self.op.add_inp_tensors(
            self.resolver.get_ir_tensor(mge_opr.inputs[1], user_opr=self.op)
        )
        weight_tensor = self.resolver.get_ir_tensor(
            mge_opr.inputs[0], user_opr=self.op, axis_order=AxisOrder.IOHW
        )
        weight_tensor.axis_order = AxisOrder.IOHW
        self.op.add_inp_tensors(weight_tensor)
        if len(mge_opr.inputs) > 2:
            self.op.add_inp_tensors(
                self.resolver.get_ir_tensor(mge_opr.inputs[2], user_opr=self.op)
            )

        for x in mge_opr.outputs:
            self.op.add_out_tensors(self.resolver.get_ir_tensor(x, owner_opr=self.op))


mode_map = {"MAX": MaxPool2dOpr, "AVERAGE_COUNT_EXCLUDE_PADDING": AvgPool2dOpr}


@_register_op("PoolingForward")
class GenPool2dOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.data_format = self.params["format"]
        self.mode = self.params["mode"]
        self.stride = [self.params["stride_h"], self.params["stride_w"]]
        self.padding = [self.params["pad_h"], self.params["pad_w"]]
        self.kernel_shape = [self.params["window_h"], self.params["window_w"]]

        self.op = mode_map[self.mode](self.kernel_shape, self.stride, self.padding)
        self.add_tensors(mge_opr)


@_register_op("ConvolutionBackwardFilter")
class GenConvolutionBackwardFilterOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        assert self.params["format"] == "NCHW", "do not support this {}".format(
            self.params["format"]
        )
        src, grad_out, weight = mge_opr.inputs

        self.ni, self.ci, self.hi, self.wi = src.shape
        self.no, self.co, self.ho, self.wo = grad_out.shape
        assert self.ni == self.no

        if len(weight.shape) == 5:
            self.group = weight.shape[0]
            self.kernel_shape = [weight.shape[3], weight.shape[4]]
        else:
            self.group = 1
            self.kernel_shape = [weight.shape[2], weight.shape[3]]

        self.stride = [self.params["stride_h"], self.params["stride_w"]]
        self.padding = [self.params["pad_h"], self.params["pad_w"]]
        self.dilation = [self.params["dilate_h"], self.params["dilate_w"]]

        self.op = ConvolutionBackwardFilterOpr(
            self.stride,
            self.padding,
            self.dilation,
            self.group,
            self.kernel_shape,
            src.shape,
            grad_out.shape,
        )
        self.add_tensors(mge_opr)


@_register_op("BatchNormForward")
class GenBatchNormalizationOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.output_idx = -1
        epsilon = self.params["epsilon"] if "epsilon" in self.params else 1e-5
        self.op = BatchNormalizationOpr(eps=epsilon, output_idx=self.output_idx)
        self.add_tensors(mge_opr)

    def add_tensors(self, mge_opr):
        # set inp var: input, scale, bias, mean, var
        for x in mge_opr.inputs:
            self.op.add_inp_tensors(self.resolver.get_ir_tensor(x, user_opr=self.op))
        # set out var
        for x in mge_opr.outputs:
            self.op.add_out_tensors(self.resolver.get_ir_tensor(x, owner_opr=self.op))
