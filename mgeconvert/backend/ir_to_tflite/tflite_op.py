# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error
import collections
from typing import List

import numpy as np
from megengine import get_logger
from numpy import dtype

from ...converter_ir.ir_op import (
    AbsOpr,
    AddOpr,
    AvgPool2dOpr,
    AxisAddRemoveOpr,
    ConcatOpr,
    Conv2dOpr,
    Deconv2dOpr,
    ExpOpr,
    FuseMulAdd3Opr,
    GetSubTensorOpr,
    LeakyReluOpr,
    LinearOpr,
    MatMulOpr,
    MaxOpr,
    MaxPool2dOpr,
    MinOpr,
    MulOpr,
    OpBase,
    PadOpr,
    PixelShuffle,
    PowOpr,
    ReduceOpr,
    ReluOpr,
    ReshapeOpr,
    ResizeOpr,
    SigmoidOpr,
    SoftmaxOpr,
    SqueezeOpr,
    SubOpr,
    TransposeOpr,
    TrueDivOpr,
    TypeCvtOpr,
)
from ...converter_ir.ir_quantizer import IRQuantizer
from ...converter_ir.ir_tensor import IOHWFormat, IRTensor, NCHWFormat, OIHWFormat
from ...converter_ir.ir_transform import cal_pad_mode
from .pyflexbuffers import dumps
from .tflite import (
    AbsOptions,
    AddOptions,
    CastOptions,
    ConcatenationOptions,
    Conv2DOptions,
    DepthToSpaceOptions,
    DepthwiseConv2DOptions,
    DivOptions,
    ExpOptions,
    FullyConnectedOptions,
    LeakyReluOptions,
    MaximumMinimumOptions,
    MulOptions,
    PadOptions,
    Pool2DOptions,
    PowOptions,
    ReducerOptions,
    ReshapeOptions,
    ResizeBilinearOptions,
    SoftmaxOptions,
    SpaceToDepthOptions,
    SqueezeOptions,
    StridedSliceOptions,
    SubOptions,
    TransposeConvOptions,
    TransposeOptions,
)
from .tflite.ActivationFunctionType import ActivationFunctionType
from .tflite.BuiltinOperator import BuiltinOperator
from .tflite.BuiltinOptions import BuiltinOptions
from .tflite.Padding import Padding
from .tflite.TensorType import TensorType

logger = get_logger(__name__)


class Config:
    platform = "official"
    require_quantize = True
    tensor_format = "nhwc"


def set_platform(platform):
    assert platform in ["official", "mtk"]
    Config.platform = platform


def set_quantization(require_quantize):
    Config.require_quantize = require_quantize


def set_tensor_format(tensor_format):
    assert tensor_format in ["nchw", "nhwc"]
    Config.tensor_format = tensor_format


def get_platform():
    return Config.platform


def get_format():
    return Config.tensor_format


def get_quantization():
    return Config.require_quantize


def _get_tensor_shape(tensor, mge_opr, disable_nhwc):
    if isinstance(mge_opr, ReshapeOpr):
        return tensor.shape

    shape = list(tensor.shape)
    if tensor.axis_order and tensor.ndim == 4:
        # OC, IC, H, W  to  OC, H, W, IC
        # NCHW to NHWC
        # except the output of reshape
        if not disable_nhwc:
            if (
                isinstance(tensor.axis_order, OIHWFormat)
                and mge_opr.name == "Conv2d"
                and mge_opr.groups > 1  # type: ignore
            ):
                # Filter in DepthwiseConv is expected to be [1, H, W, O].
                shape = tensor.axis_order.shape_to_IHWO(shape)
            elif isinstance(tensor.axis_order, NCHWFormat):
                shape = tensor.axis_order.shape_to_NHWC(shape)
            elif isinstance(tensor.axis_order, IOHWFormat):
                shape = tensor.axis_order.shape_to_OHWI(shape)
    elif tensor.axis_order and mge_opr.name == "Squeeze":
        if not disable_nhwc:
            nhwc_aixs_order = [0, 3, 1, 2]
            inp_shape = list(mge_opr.inp_tensors[0].shape)
            assert len(inp_shape) == 4
            out_shape = mge_opr.inp_tensors[0].axis_order.shape_to_NHWC(inp_shape)
            squeeze_dims = [nhwc_aixs_order[i] for i in mge_opr.squeeze_dims[::-1]]
            for i in squeeze_dims:
                out_shape.pop(i)
            shape = out_shape

    elif tensor.ndim > 4:
        assert False, "ERROR: output ndim {0} is not supported now".format(tensor.ndim)
    return shape


def _get_tensor_value(tensor, mge_opr, quantizer, disable_nhwc):
    if isinstance(mge_opr, ReshapeOpr):
        return None
    number_list: List[np.ndarray] = []
    if (
        quantizer.require_quantize
        and hasattr(tensor, "scale")
        and tensor.np_data is not None
    ):
        value = quantizer.quantize(tensor)
    else:
        value = tensor.np_data
    if value is not None:
        if not disable_nhwc and tensor.axis_order and value.ndim == 4:
            if (
                isinstance(tensor.axis_order, OIHWFormat)
                and mge_opr.name == "Conv2d"
                and mge_opr.groups > 1  # type: ignore
            ):
                # Filter in DepthwiseConv is expected to be [1, H, W, O].
                value = tensor.axis_order.data_to_IHWO(value)
            elif isinstance(tensor.axis_order, NCHWFormat):
                value = tensor.axis_order.data_to_NHWC(value)
            elif isinstance(tensor.axis_order, IOHWFormat):
                value = tensor.axis_order.data_to_OHWI(value)

        if not disable_nhwc and mge_opr.name == "GetSubTensor" and value is not None:
            assert value.shape == (
                4,
            ), "can't support Slice input ndim !=4 in nhwc mode "
            value = np.array([value[0], value[2], value[3], value[1]])
        number_list = value.reshape(-1)

    if len(number_list) > 0:
        byte_list: List[bytes] = []
        for i in number_list:
            byte_list.extend(i.tobytes())
        return byte_list
    else:
        return None


def get_shape_param(
    tensor: IRTensor, mge_opr: OpBase, quantizer: IRQuantizer, disable_nhwc=False
):
    """
    Return a tuple of shape and bytes(1dim) object for tflite operator, which will
    restore its inp/out at runtime by the shape and bytes.
    """
    return (
        _get_tensor_shape(tensor, mge_opr, disable_nhwc),
        _get_tensor_value(tensor, mge_opr, quantizer, disable_nhwc),
    )


mge2tflite_dtype_mapping = {
    # pylint: disable=no-member
    np.float32: TensorType.FLOAT32,
    np.float16: TensorType.FLOAT16,
    np.int32: TensorType.INT32,
    np.int16: TensorType.INT16,
    np.int8: TensorType.INT8,
    np.uint8: TensorType.UINT8,
    dtype("int32"): TensorType.INT32,
    dtype("int16"): TensorType.INT16,
    dtype("uint8"): TensorType.UINT8,
    dtype("int8"): TensorType.INT8,
    "quint8": TensorType.UINT8,
    "qint8": TensorType.INT8,
    "qint32": TensorType.INT32,
    "qint16": TensorType.INT16,
    "uint8": TensorType.UINT8,
    "int8": TensorType.INT8,
    "int16": TensorType.INT16,
    "int32": TensorType.INT32,
    "qint8_narrow": TensorType.INT8,
}


mge2tflite_activation_type = {
    "IDENTITY": ActivationFunctionType.NONE,
    "RELU": ActivationFunctionType.RELU,
    "RELU6": ActivationFunctionType.RELU6,
    "TANH": ActivationFunctionType.TANH,
}


MGE2TFLITE = {}


def _register_op(*ops):
    def callback(impl):
        for op in ops:
            MGE2TFLITE[op] = impl

    return callback


@_register_op(AddOpr, FuseMulAdd3Opr)
def _add(mge_opr, builder):  # pylint: disable=too-many-return-statements
    AddOptions.AddOptionsStart(builder)
    AddOptions.AddOptionsAddFusedActivationFunction(
        builder, mge2tflite_activation_type[mge_opr.activation]
    )
    options = AddOptions.AddOptionsEnd(builder)
    return BuiltinOperator.ADD, BuiltinOptions.AddOptions, options


@_register_op(SubOpr)
def _sub(mge_opr, builder):
    SubOptions.SubOptionsStart(builder)
    SubOptions.SubOptionsAddFusedActivationFunction(
        builder, mge2tflite_activation_type[mge_opr.activation]
    )
    options = SubOptions.SubOptionsEnd(builder)
    return BuiltinOperator.SUB, BuiltinOptions.SubOptions, options


@_register_op(AbsOpr)
def _abs(_, builder):
    AbsOptions.AbsOptionsStart(builder)
    options = AbsOptions.AbsOptionsEnd(builder)
    return BuiltinOperator.ABS, BuiltinOptions.AbsOptions, options


@_register_op(SigmoidOpr)
def _sigmoid(*_):  # pylint: disable=too-many-return-statements
    return BuiltinOperator.LOGISTIC, None, None


@_register_op(ReluOpr)
def _relu(*_):
    return BuiltinOperator.RELU, None, None


@_register_op(MulOpr)
def _mul(mge_opr, builder):  # pylint: disable=too-many-return-statements
    MulOptions.MulOptionsStart(builder)
    MulOptions.MulOptionsAddFusedActivationFunction(
        builder, mge2tflite_activation_type[mge_opr.activation]
    )
    options = MulOptions.MulOptionsEnd(builder)
    return BuiltinOperator.MUL, BuiltinOptions.MulOptions, options


@_register_op(TrueDivOpr)
def _div(mge_opr, builder):
    DivOptions.DivOptionsStart(builder)
    DivOptions.DivOptionsAddFusedActivationFunction(
        builder, mge2tflite_activation_type[mge_opr.activation]
    )
    options = DivOptions.DivOptionsEnd(builder)
    return BuiltinOperator.DIV, BuiltinOptions.DivOptions, options


def _padding_mode_conv(mge_opr):
    if cal_pad_mode(mge_opr) == "VALID":
        return Padding.VALID
    else:
        return Padding.SAME


@_register_op(Conv2dOpr)
def _conv2d(mge_opr, builder):
    if mge_opr.groups > 1:
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsStart(builder)
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPadding(
            builder, _padding_mode_conv(mge_opr)
        )
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideH(
            builder, mge_opr.stride[0]
        )
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideW(
            builder, mge_opr.stride[1]
        )
        assert isinstance(mge_opr.inp_tensors[1].axis_order, OIHWFormat)
        assert isinstance(mge_opr.inp_tensors[0].axis_order, NCHWFormat)
        num_filter_channels = mge_opr.inp_tensors[1].shape[0]
        num_input_channels = mge_opr.inp_tensors[0].shape[1]
        assert num_filter_channels % num_input_channels == 0
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDepthMultiplier(
            builder, num_filter_channels // num_input_channels
        )
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddFusedActivationFunction(
            builder, mge2tflite_activation_type[mge_opr.activation]
        )
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationHFactor(
            builder, mge_opr.dilation[0]
        )
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationWFactor(
            builder, mge_opr.dilation[1]
        )
        options = DepthwiseConv2DOptions.DepthwiseConv2DOptionsEnd(builder)
        return (
            BuiltinOperator.DEPTHWISE_CONV_2D,
            BuiltinOptions.DepthwiseConv2DOptions,
            options,
        )

    Conv2DOptions.Conv2DOptionsStart(builder)
    Conv2DOptions.Conv2DOptionsAddPadding(builder, _padding_mode_conv(mge_opr))
    Conv2DOptions.Conv2DOptionsAddStrideH(builder, mge_opr.stride[0])
    Conv2DOptions.Conv2DOptionsAddStrideW(builder, mge_opr.stride[1])
    Conv2DOptions.Conv2DOptionsAddDilationHFactor(builder, mge_opr.dilation[0])
    Conv2DOptions.Conv2DOptionsAddDilationWFactor(builder, mge_opr.dilation[1])
    Conv2DOptions.Conv2DOptionsAddFusedActivationFunction(
        builder, mge2tflite_activation_type[mge_opr.activation]
    )
    options = Conv2DOptions.Conv2DOptionsEnd(builder)
    return BuiltinOperator.CONV_2D, BuiltinOptions.Conv2DOptions, options


@_register_op(PadOpr)
def _pad(_, builder):
    PadOptions.PadOptionsStart(builder)
    options = PadOptions.PadOptionsEnd(builder)
    return BuiltinOperator.PAD, BuiltinOptions.PadOptions, options


def _padding_mode_transpose_conv(mge_opr):
    if (
        mge_opr.out_tensors[0].shape[3]
        == mge_opr.inp_tensors[2].shape[3] * mge_opr.stride[0]
        and mge_opr.out_tensors[0].shape[2]
        == mge_opr.inp_tensors[2].shape[2] * mge_opr.stride[1]
    ):
        # padding mode == SAME
        return Padding.SAME
    elif mge_opr.padding[0] == 0 and mge_opr.padding[1] == 0:
        # padding mode == VALID
        return Padding.VALID
    else:
        assert False, "ERROR: unsupported padding mode"
        return None


@_register_op(Deconv2dOpr)
def _deconv(mge_opr, builder):
    if get_platform() == "mtk":
        CustomOperator = collections.namedtuple("CustomOperator", ["type", "code"])
        CustomOptions = collections.namedtuple("CustomOptions", ["code"])

        options = dict()
        options["PaddingType"] = _padding_mode_transpose_conv(mge_opr)
        options["stride_height"] = mge_opr.stride[0]
        options["stride_width"] = mge_opr.stride[1]
        options["depth_multiplier"] = 1
        options["dilation_height_factor"] = mge_opr.dilation[0]
        options["dilation_width_factor"] = mge_opr.dilation[1]
        options["activation"] = mge2tflite_activation_type[mge_opr.activation]
        return (
            CustomOperator(type=BuiltinOperator.CUSTOM, code="MTK_TRANSPOSE_CONV"),
            CustomOptions("MTK_TRANSPOSE_CONV"),
            dumps(options),
        )

    TransposeConvOptions.TransposeConvOptionsStart(builder)
    TransposeConvOptions.TransposeConvOptionsAddPadding(
        builder, _padding_mode_transpose_conv(mge_opr)
    )
    TransposeConvOptions.TransposeConvOptionsAddStrideH(builder, mge_opr.stride[0])
    TransposeConvOptions.TransposeConvOptionsAddStrideW(builder, mge_opr.stride[1])
    options = TransposeConvOptions.TransposeConvOptionsEnd(builder)
    return BuiltinOperator.TRANSPOSE_CONV, BuiltinOptions.TransposeConvOptions, options


@_register_op(ConcatOpr)
def _concat(mge_opr, builder):
    if (
        mge_opr.inp_tensors[0].q_dtype == "int8"
        and len({t.scale for t in mge_opr.inp_tensors + mge_opr.out_tensors}) != 1
    ):
        logger.warning(
            "tflite int8 concat doesn't support inputs outputs with different scale!"
        )
    if mge_opr.inp_tensors[0].q_dtype == "int16" and not all(
        [t.zero_point == 0 for t in mge_opr.inp_tensors + mge_opr.out_tensors]
    ):
        logger.warning(
            "tflite int16 concat doesn't support inputs outputs with zero point != 0!"
        )

    ConcatenationOptions.ConcatenationOptionsStart(builder)
    ConcatenationOptions.ConcatenationOptionsAddFusedActivationFunction(
        builder, mge2tflite_activation_type[mge_opr.activation]
    )
    axis = mge_opr.axis
    if mge_opr.inp_tensors[0].ndim == 4:
        # map NCHW to NHWC
        if mge_opr.axis == 1:
            axis = 3
        elif mge_opr.axis == 2:
            axis = 1
        elif mge_opr.axis == 3:
            axis = 2
    ConcatenationOptions.ConcatenationOptionsAddAxis(builder, axis)
    options = ConcatenationOptions.ConcatenationOptionsEnd(builder)
    return BuiltinOperator.CONCATENATION, BuiltinOptions.ConcatenationOptions, options


@_register_op(PowOpr)
def _pow(_, builder):
    PowOptions.PowOptionsStart(builder)
    options = PowOptions.PowOptionsEnd(builder)
    return BuiltinOperator.POW, BuiltinOptions.PowOptions, options


@_register_op(ExpOpr)
def _exp(_, builder):
    ExpOptions.ExpOptionsStart(builder)
    options = ExpOptions.ExpOptionsEnd(builder)
    return BuiltinOperator.EXP, BuiltinOptions.ExpOptions, options


@_register_op(MaxOpr)
def _max(_, builder):
    MaximumMinimumOptions.MaximumMinimumOptionsStart(builder)
    options = MaximumMinimumOptions.MaximumMinimumOptionsEnd(builder)
    return BuiltinOperator.MAXIMUM, BuiltinOptions.MaximumMinimumOptions, options


@_register_op(MinOpr)
def _min(_, builder):
    MaximumMinimumOptions.MaximumMinimumOptionsStart(builder)
    options = MaximumMinimumOptions.MaximumMinimumOptionsEnd(builder)
    return BuiltinOperator.MINIMUM, BuiltinOptions.MaximumMinimumOptions, options


@_register_op(ReshapeOpr, AxisAddRemoveOpr)
def _reshape(mge_opr, builder):
    ReshapeOptions.ReshapeOptionsStartNewShapeVector(builder, len(mge_opr.out_shape))
    for i in reversed(list(mge_opr.out_shape)):
        builder.PrependInt32(i)
    new_shape = builder.EndVector(len(mge_opr.out_shape))
    ReshapeOptions.ReshapeOptionsStart(builder)
    ReshapeOptions.ReshapeOptionsAddNewShape(builder, new_shape)
    options = ReshapeOptions.ReshapeOptionsEnd(builder)
    return BuiltinOperator.RESHAPE, BuiltinOptions.ReshapeOptions, options


@_register_op(ResizeOpr)
def _resize(mge_opr, builder):
    assert mge_opr.mode == "bilinear", "Resize mode should be BILINEAR."
    ResizeBilinearOptions.ResizeBilinearOptionsStart(builder)
    ResizeBilinearOptions.ResizeBilinearOptionsAddAlignCorners(builder, False)
    ResizeBilinearOptions.ResizeBilinearOptionsAddHalfPixelCenters(builder, True)
    options = ResizeBilinearOptions.ResizeBilinearOptionsEnd(builder)
    return (
        BuiltinOperator.RESIZE_BILINEAR,
        BuiltinOptions.ResizeBilinearOptions,
        options,
    )


@_register_op(ReduceOpr)
def _reduce(mge_opr, builder):
    ReducerOptions.ReducerOptionsStart(builder)
    ReducerOptions.ReducerOptionsAddKeepDims(builder, False)
    options = ReducerOptions.ReducerOptionsEnd(builder)

    reduce_mode_map = {
        "SUM": BuiltinOperator.SUM,
        "MEAN": BuiltinOperator.MEAN,
        "MAX": BuiltinOperator.REDUCE_MAX,
        "MIN": BuiltinOperator.REDUCE_MIN,
    }
    return reduce_mode_map[mge_opr.mode], BuiltinOptions.ReducerOptions, options


@_register_op(GetSubTensorOpr)
def _getsubtensor(_, builder):
    StridedSliceOptions.StridedSliceOptionsStart(builder)
    options = StridedSliceOptions.StridedSliceOptionsEnd(builder)
    return BuiltinOperator.STRIDED_SLICE, BuiltinOptions.StridedSliceOptions, options


@_register_op(MaxPool2dOpr, AvgPool2dOpr)
def _pooling(mge_opr, builder):
    Pool2DOptions.Pool2DOptionsStart(builder)
    Pool2DOptions.Pool2DOptionsAddPadding(builder, Padding.VALID)
    shape = mge_opr.inp_tensors[0].shape
    if (
        get_platform() == "mtk"
        and shape[2] == mge_opr.kernel_size[0]
        and shape[3] == mge_opr.kernel_size[1]
    ):
        # MTK global pooling
        print(
            "\nWARNING: the stride of global pooling "
            "would be changed to 1 to adapted the bug of MTK"
        )
        Pool2DOptions.Pool2DOptionsAddStrideH(builder, 1)
        Pool2DOptions.Pool2DOptionsAddStrideW(builder, 1)
    else:
        Pool2DOptions.Pool2DOptionsAddStrideH(builder, mge_opr.stride[0])
        Pool2DOptions.Pool2DOptionsAddStrideW(builder, mge_opr.stride[1])
    Pool2DOptions.Pool2DOptionsAddFilterHeight(builder, mge_opr.kernel_size[0])
    Pool2DOptions.Pool2DOptionsAddFilterWidth(builder, mge_opr.kernel_size[1])
    Pool2DOptions.Pool2DOptionsAddFusedActivationFunction(
        builder, mge2tflite_activation_type[mge_opr.activation]
    )
    options = Pool2DOptions.Pool2DOptionsEnd(builder)

    tfl_opr_type = BuiltinOperator.AVERAGE_POOL_2D
    if mge_opr.name == "MaxPool2d":
        tfl_opr_type = BuiltinOperator.MAX_POOL_2D
    return tfl_opr_type, BuiltinOptions.Pool2DOptions, options


@_register_op(MatMulOpr, LinearOpr)
def _matrix_mul(mge_opr, builder):
    FullyConnectedOptions.FullyConnectedOptionsStart(builder)
    # mge quantized model should not have bias for tflite conversion
    FullyConnectedOptions.FullyConnectedOptionsAddFusedActivationFunction(
        builder, mge2tflite_activation_type[mge_opr.activation]
    )
    options = FullyConnectedOptions.FullyConnectedOptionsEnd(builder)
    return (
        BuiltinOperator.FULLY_CONNECTED,
        BuiltinOptions.FullyConnectedOptions,
        options,
    )


@_register_op(SqueezeOpr)
def _squeeze(mge_opr, builder):
    SqueezeOptions.SqueezeOptionsStartSqueezeDimsVector(
        builder, len(mge_opr.squeeze_dims)
    )
    if get_format() == "nhwc":
        assert (
            mge_opr.inp_tensors[0].ndim == 4
        ), "can't support Squeeze input ndim !=4 in nhwc mode"
        nhwc_aixs_order = [0, 3, 1, 2]
        squeeze_dims = [nhwc_aixs_order[i] for i in mge_opr.squeeze_dims]
    else:
        squeeze_dims = mge_opr.squeeze_dims
    for i in squeeze_dims:
        builder.PrependInt32(i)
    squeeze_dims = builder.EndVector(len(squeeze_dims))
    SqueezeOptions.SqueezeOptionsStart(builder)
    SqueezeOptions.SqueezeOptionsAddSqueezeDims(builder, squeeze_dims)
    options = SqueezeOptions.SqueezeOptionsEnd(builder)
    return BuiltinOperator.SQUEEZE, BuiltinOptions.SqueezeOptions, options


@_register_op(TransposeOpr)
def _transpose(_, builder):
    TransposeOptions.TransposeOptionsStart(builder)
    options = TransposeOptions.TransposeOptionsEnd(builder)
    return BuiltinOperator.TRANSPOSE, BuiltinOptions.TransposeOptions, options


@_register_op(SoftmaxOpr)
def _softmax(mge_opr, builder):
    SoftmaxOptions.SoftmaxOptionsStart(builder)
    SoftmaxOptions.SoftmaxOptionsAddBeta(builder, mge_opr.beta)
    options = SoftmaxOptions.SoftmaxOptionsEnd(builder)
    return BuiltinOperator.SOFTMAX, BuiltinOptions.SoftmaxOptions, options


@_register_op(LeakyReluOpr)
def _leaky_relu(mge_opr, builder):
    LeakyReluOptions.LeakyReluOptionsStart(builder)
    LeakyReluOptions.LeakyReluOptionsAddAlpha(builder, mge_opr.negative_slope)
    options = LeakyReluOptions.LeakyReluOptionsEnd(builder)
    return BuiltinOperator.LEAKY_RELU, BuiltinOptions.LeakyReluOptions, options


@_register_op(TypeCvtOpr)
def _typecvt(mge_opr, builder):
    if get_quantization():
        target_type = mge_opr.inp_tensors[0].q_dtype
    else:
        target_type = mge_opr.inp_tensors[0].dtype

    CastOptions.CastOptionsStart(builder)
    CastOptions.CastOptionsAddInDataType(builder, mge2tflite_dtype_mapping[target_type])
    CastOptions.CastOptionsAddOutDataType(
        builder, mge2tflite_dtype_mapping[target_type]
    )
    options = CastOptions.CastOptionsEnd(builder)
    return BuiltinOperator.CAST, BuiltinOptions.CastOptions, options


@_register_op(PixelShuffle)
def _pixel_shuffle(mge_opr, builder):
    op_map = {
        "upsample": BuiltinOperator.DEPTH_TO_SPACE,
        "downsample": BuiltinOperator.SPACE_TO_DEPTH,
    }
    option_map = {
        "upsample": BuiltinOptions.DepthToSpaceOptions,
        "downsample": BuiltinOptions.SpaceToDepthOptions,
    }
    if mge_opr.mode == "upsample":
        DepthToSpaceOptions.DepthToSpaceOptionsStart(builder)
        DepthToSpaceOptions.DepthToSpaceOptionsAddBlockSize(
            builder, mge_opr.scale_factor
        )
        options = DepthToSpaceOptions.DepthToSpaceOptionsEnd(builder)
    elif mge_opr.mode == "downsample":
        SpaceToDepthOptions.SpaceToDepthOptionsStart(builder)
        SpaceToDepthOptions.SpaceToDepthOptionsAddBlockSize(
            builder, mge_opr.scale_factor
        )
        options = SpaceToDepthOptions.SpaceToDepthOptionsEnd(builder)
    return (
        op_map[mge_opr.mode],
        option_map[mge_opr.mode],
        options,
    )
