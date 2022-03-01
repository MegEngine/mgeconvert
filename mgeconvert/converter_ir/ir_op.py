# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import List  # pylint: disable=unused-import

from .ir_tensor import IRTensor  # pylint: disable=unused-import


class OpBase:
    skip = False
    name = ""

    def __init__(self) -> None:
        self.inp_tensors = []  # type: List[IRTensor]
        self.out_tensors = []  # type: List[IRTensor]
        self.activation = "IDENTITY"

    def add_inp_tensors(self, ir_tensor):
        self.inp_tensors.append(ir_tensor)

    def add_out_tensors(self, ir_tensor):
        self.out_tensors.append(ir_tensor)


####################  conv related ############################
class _ConvOpr(OpBase):
    def __init__(self, stride, padding, dilation, groups):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups


class Conv2dOpr(_ConvOpr):
    name = "Conv2d"

    def __init__(self, stride, padding, dilation, groups, auto_pad="NOTSET"):
        super().__init__(stride, padding, dilation, groups)
        self.auto_pad = auto_pad


class Deconv2dOpr(_ConvOpr):
    name = "Deconv2d"


class ConvRelu2dOpr(_ConvOpr):
    name = "ConvRelu2d"


class ConvolutionBackwardFilterOpr(OpBase):
    name = "ConvolutionBackwardFilter"

    def __init__(
        self, stride, padding, dilation, group, kernel_shape, src_shape, grad_out_shape
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.kernel_shape = kernel_shape
        self.src_shape = src_shape
        self.grad_out_shape = grad_out_shape


class _PoolOpr(OpBase):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


class MaxPool2dOpr(_PoolOpr):
    name = "MaxPool2d"

    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        auto_pad="NOTSET",
        ceil_mode=0,
        dilations=None,
        storage_order=0,
    ):
        super().__init__(kernel_size, stride, padding)
        self.auto_pad = auto_pad
        self.ceil_mode = ceil_mode
        self.dilations = dilations
        self.storage_order = storage_order
        self.mode = "MAX"


class AvgPool2dOpr(_PoolOpr):
    name = "AvgPool2d"

    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        mode="AVERAGE_COUNT_EXCLUDE_PADDING",
        auto_pad="NOTSET",
        ceil_mode=0,
    ):
        super().__init__(kernel_size, stride, padding)
        self.mode = mode
        self.auto_pad = auto_pad
        self.ceil_mode = ceil_mode


class PadOpr(OpBase):
    name = "Pad"

    def __init__(self, pad_width=None, mode="constant", pad_val=0.0):
        super().__init__()
        self.pad_width = pad_width
        self.mode = mode
        self.pad_val = pad_val


class BatchNormalizationOpr(OpBase):
    name = "BatchNormalization"

    def __init__(
        self,
        weight=None,
        bias=None,
        mean=None,
        var=None,
        eps=1e-5,
        momentum=0.9,
        output_idx=-1,
    ):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.mean = mean
        self.var = var
        self.eps = eps
        self.momentum = momentum
        self.output_idx = output_idx


class AdaptiveAvgPool2dOpr(OpBase):
    name = "AdaptiveAvgPool2d"

    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape


####################  math related ############################


class MatMulOpr(OpBase):
    name = "MatMul"

    def __init__(
        self,
        transpose_a=False,
        transpose_b=False,
        compute_mode="default",
        format="default",
        alpha=1.0,
        beta=1.0,
    ):
        super().__init__()
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.compute_mode = compute_mode
        self.format = format
        self.alpha = alpha
        self.beta = beta


class LinearOpr(MatMulOpr):
    name = "Linear"

    def __init__(self, has_bias=False):
        super().__init__(transpose_b=True)
        self.has_bias = has_bias


class ReduceOpr(OpBase):
    name = "Reduce"

    def __init__(self, axis, mode, keep_dims):
        super().__init__()
        self.axis = axis
        self.mode = mode
        self.keep_dims = keep_dims


class SoftmaxOpr(OpBase):
    name = "Softmax"

    def __init__(self, axis=None, beta=1):
        super().__init__()
        self.axis = axis
        self.beta = beta


class ResizeOpr(OpBase):
    name = "Resize"

    def __init__(
        self, out_size, scale_factor=None, mode="bilinear", align_corners=None
    ):
        super().__init__()
        self.out_size = out_size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.extra_param = {}


####################  tensor related ############################
class FlattenOpr(OpBase):
    name = "Flatten"

    def __init__(self, start_axis=0, end_axis=-1):
        super().__init__()
        self.start_axis = start_axis
        self.end_axis = end_axis


class DropoutOpr(OpBase):
    name = "Dropout"

    def __init__(self, drop_prob=0, training=False):
        super().__init__()
        self.drop_prob = drop_prob
        self.training = training


class ConstantOpr(OpBase):
    name = "Constant"


class MultipleDeviceTensorHolderOpr(OpBase):
    name = "MultipleDeviceTensorHolder"


class SharedDeviceTensorOpr(OpBase):
    name = "SharedDeviceTensorOpr"


class VolatileSharedDeviceTensorOpr(OpBase):
    name = "VolatileSharedDeviceTensor"


class GetVarShapeOpr(OpBase):
    name = "GetVarShape"


class IndexingOneHotOpr(OpBase):
    name = "IndexingOneHotOpr"


class LinspaceOpr(OpBase):
    name = "Linspace"


class WarpPerspectiveForwardOpr(OpBase):
    name = "WarpPerspectiveForward"


class IdentityOpr(OpBase):
    name = "Identity"

    def __init__(self):
        super().__init__()
        self.mode = "Identity"


class ConcatOpr(OpBase):
    name = "Concat"

    def __init__(self, axis):
        super().__init__()
        self.axis = axis


class ReshapeOpr(OpBase):
    name = "Reshape"

    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape
        self.allowzero = 0


class TransposeOpr(OpBase):
    name = "Transpose"

    def __init__(self, pattern: list):
        super().__init__()
        self.pattern = pattern


class SqueezeOpr(OpBase):
    name = "Squeeze"

    def __init__(self, squeeze_dims):
        super().__init__()
        self.squeeze_dims = squeeze_dims


class GetSubTensorOpr(OpBase):
    name = "GetSubTensor"

    def __init__(self, axis, begin_params, end_params, step_params, squeeze_axis=None):
        super().__init__()
        self.axis = axis
        self.begin_params = begin_params
        self.end_params = end_params
        self.step_params = step_params
        self.squeeze_axis = squeeze_axis


class GatherOpr(OpBase):
    name = "Gather"

    def __init__(self, axis):
        super().__init__()
        self.axis = axis


class AxisAddRemoveOpr(OpBase):
    name = "AxisAddRemove"

    def __init__(self, out_shape, desc):
        super().__init__()
        self.out_shape = out_shape
        self.desc = desc


class BroadcastOpr(OpBase):
    name = "Broadcast"


class PixelShuffle(OpBase):
    name = "PixelShuffle"

    def __init__(self, scale_factor, mode="downsample"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode


############################ elemwise ########################


class ElemwiseOpr(OpBase):
    pass


class AddOpr(ElemwiseOpr):
    name = "Add"


class SubOpr(ElemwiseOpr):
    name = "Sub"


class MulOpr(ElemwiseOpr):
    name = "Mul"


class TrueDivOpr(ElemwiseOpr):
    name = "TrueDiv"


class PowOpr(ElemwiseOpr):
    name = "Pow"


class ExpOpr(ElemwiseOpr):
    name = "Exp"


class FloorOpr(ElemwiseOpr):
    name = "Floor"


class FloorDivOpr(ElemwiseOpr):
    name = "FloorDiv"


class CeilOpr(ElemwiseOpr):
    name = "Ceil"


class MaxOpr(ElemwiseOpr):
    name = "Max"


class MinOpr(ElemwiseOpr):
    name = "Min"


class AbsOpr(ElemwiseOpr):
    name = "Abs"


class LogOpr(ElemwiseOpr):
    name = "Log"


class FuseMulAdd3Opr(OpBase):
    name = "FuseMulAdd3"


class FuseAddReluOpr(OpBase):
    name = "FuseAddRelu"


############################# activation ###########################


class Relu6Opr(OpBase):
    name = "Relu6"


class ReluOpr(OpBase):
    name = "Relu"


class SigmoidOpr(OpBase):
    name = "Sigmoid"


class HardSigmoidOpr(OpBase):
    name = "HardSigmoid"


class SiLUOpr(OpBase):
    name = "SiLU"


class TanHOpr(OpBase):
    name = "TanH"


class LeakyReluOpr(OpBase):
    name = "LeakyRelu"

    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope


class TypeCvtOpr(OpBase):
    name = "TypeCvt"

    def __init__(self, out_dtype):
        super().__init__()
        self.out_dtype = out_dtype


class HardSwishOpr(OpBase):
    name = "HardSwish"


class RepeatOpr(OpBase):
    name = "Repeat"

    def __init__(self, repeats, axis):
        super().__init__()
        self.repeats = repeats
        self.axis = 0 if axis is None else axis


class ClipOpr(OpBase):
    name = "Clip"

    def __init__(self, upper=float("inf"), lower=float("-inf")):
        super().__init__()
        self.upper = upper
        self.lower = lower


class LstmOpr(OpBase):
    name = "LSTM"

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        direction,
        proj_size,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.direction = direction
        self.proj_size = proj_size
        self.batch_size = 1

        self.activation_alpha = None
        self.activation_beta = None
        self.activations = None
        self.clip = None
        self.input_forget = None
        self.output_sequence = None
        self.weight_ih_l = []
        self.weight_ih_l_reverse = []
        self.weight_hh_l = []
        self.weight_hh_l_reverse = []
        self.bias_ih_l = []
        self.bias_ih_l_reverse = []
        self.bias_hh_l = []
        self.bias_hh_l_reverse = []
        self.sequence_lens = None
        self.p = None
