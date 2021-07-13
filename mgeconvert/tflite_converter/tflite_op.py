# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections

import numpy as np
from numpy import dtype

from ..mge_context import (
    AxisAddRemoveOpr,
    ConcatOpr,
    ConvBiasForwardOpr,
    ConvolutionBackwardDataOpr,
    ConvolutionForwardOpr,
    DimshuffleOpr,
    ElemwiseMultiTypeOpr,
    ElemwiseOpr,
    MatrixMulOpr,
    PadOpr,
    PoolingForwardOpr,
    ReduceOpr,
    ReshapeOpr,
    ResizeForwardOpr,
    SoftmaxOpr,
    get_platform,
)
from .pyflexbuffers import dumps  # pylint: disable=import-error
from .tflite import (  # pylint: disable=import-error
    AddOptions,
    ConcatenationOptions,
    Conv2DOptions,
    DepthwiseConv2DOptions,
    DivOptions,
    ExpOptions,
    FullyConnectedOptions,
    MaximumMinimumOptions,
    MulOptions,
    NegOptions,
    PadOptions,
    Pool2DOptions,
    ReducerOptions,
    ReshapeOptions,
    ResizeBilinearOptions,
    SoftmaxOptions,
    SubOptions,
    TransposeConvOptions,
    TransposeOptions,
)
from .tflite.ActivationFunctionType import (  # pylint: disable=import-error
    ActivationFunctionType,
)
from .tflite.BuiltinOperator import BuiltinOperator  # pylint: disable=import-error
from .tflite.BuiltinOptions import BuiltinOptions  # pylint: disable=import-error
from .tflite.Padding import Padding  # pylint: disable=import-error
from .tflite.TensorType import TensorType  # pylint: disable=import-error


def get_shape_param(tensor, mge_opr, disable_nhwc=False):
    if isinstance(mge_opr, ReshapeOpr):
        return tensor.shape, None

    shape = list(tensor.shape)
    if tensor.ndim == 4:
        # OC, IC, H, W  to  OC, H, W, IC
        # NCHW to NHWC
        # except the output of reshape
        if not disable_nhwc:
            if (
                hasattr(mge_opr, "type")
                and mge_opr.type == "ConvolutionBackwardData"
                and tensor.np_data is not None
            ):
                shape = [
                    tensor.shape[1],
                    tensor.shape[2],
                    tensor.shape[3],
                    tensor.shape[0],
                ]
            else:
                shape = [
                    tensor.shape[0],
                    tensor.shape[2],
                    tensor.shape[3],
                    tensor.shape[1],
                ]
    elif tensor.ndim > 4:
        assert False, "ERROR: output ndim {0} is not supported now".format(tensor.ndim)

    if tensor.is_faked:
        return shape, tensor.byte_list

    number_list = []
    value = tensor.np_data
    if value is not None:
        if value.ndim == 4:
            if hasattr(mge_opr, "type") and mge_opr.type == "ConvolutionBackwardData":
                value = value.transpose(1, 2, 3, 0)
            else:
                value = value.transpose(0, 2, 3, 1)
        number_list = value.reshape(-1)

    if len(number_list) > 0:
        byte_list = []
        for i in number_list:
            byte_list.extend(i.tobytes())
        return shape, byte_list
    else:
        return shape, None


mge2tflite_dtype_mapping = {
    # pylint: disable=no-member
    np.float32: TensorType.FLOAT32,
    np.float16: TensorType.FLOAT16,
    np.int32: TensorType.INT32,
    np.int8: TensorType.INT8,
    np.uint8: TensorType.UINT8,
    dtype("int32"): TensorType.INT32,
    dtype("uint8"): TensorType.UINT8,
    dtype("int8"): TensorType.INT8,
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


@_register_op(ElemwiseOpr, ElemwiseMultiTypeOpr)
def _elemwise(mge_opr, builder):  # pylint: disable=too-many-return-statements
    if isinstance(mge_opr, ElemwiseMultiTypeOpr):
        # TODO: currently only support ADD + RELU
        AddOptions.AddOptionsStart(builder)
        AddOptions.AddOptionsAddFusedActivationFunction(
            builder, mge2tflite_activation_type["RELU"]
        )
        options = AddOptions.AddOptionsEnd(builder)
        return BuiltinOperator.ADD, BuiltinOptions.AddOptions, options

    # return tuple of (tfl_op_type, option type, option)
    if mge_opr.mode == "NEG":
        NegOptions.NegOptionsStart(builder)
        options = NegOptions.NegOptionsEnd(builder)
        return BuiltinOperator.NEG, BuiltinOptions.NegOptions, options
    if mge_opr.mode in ("ADD", "FUSE_ADD_RELU"):
        AddOptions.AddOptionsStart(builder)
        AddOptions.AddOptionsAddFusedActivationFunction(
            builder, mge2tflite_activation_type[mge_opr.activation]
        )
        options = AddOptions.AddOptionsEnd(builder)
        return BuiltinOperator.ADD, BuiltinOptions.AddOptions, options
    if mge_opr.mode == "SUB":
        SubOptions.SubOptionsStart(builder)
        SubOptions.SubOptionsAddFusedActivationFunction(
            builder, mge2tflite_activation_type[mge_opr.activation]
        )
        options = SubOptions.SubOptionsEnd(builder)
        return BuiltinOperator.SUB, BuiltinOptions.SubOptions, options
    if mge_opr.mode == "MUL":
        MulOptions.MulOptionsStart(builder)
        MulOptions.MulOptionsAddFusedActivationFunction(
            builder, mge2tflite_activation_type[mge_opr.activation]
        )
        options = MulOptions.MulOptionsEnd(builder)
        return BuiltinOperator.MUL, BuiltinOptions.MulOptions, options
    if mge_opr.mode in ("DIV", "TRUE_DIV"):
        DivOptions.DivOptionsStart(builder)
        DivOptions.DivOptionsAddFusedActivationFunction(
            builder, mge2tflite_activation_type[mge_opr.activation]
        )
        options = DivOptions.DivOptionsEnd(builder)
        return BuiltinOperator.DIV, BuiltinOptions.DivOptions, options
    if mge_opr.mode == "EXP":
        ExpOptions.ExpOptionsStart(builder)
        options = ExpOptions.ExpOptionsEnd(builder)
        return BuiltinOperator.EXP, BuiltinOptions.ExpOptions, options
    if mge_opr.mode == "MAX":
        MaximumMinimumOptions.MaximumMinimumOptionsStart(builder)
        options = MaximumMinimumOptions.MaximumMinimumOptionsEnd(builder)
        return BuiltinOperator.MAXIMUM, BuiltinOptions.MaximumMinimumOptions, options
    if mge_opr.mode == "MIN":
        MaximumMinimumOptions.MaximumMinimumOptionsStart(builder)
        options = MaximumMinimumOptions.MaximumMinimumOptionsEnd(builder)
        return BuiltinOperator.MINIMUM, BuiltinOptions.MaximumMinimumOptions, options
    if mge_opr.mode == "SIGMOID":
        return BuiltinOperator.LOGISTIC, None, None
    return None, None, None


@_register_op(ReduceOpr)
def _reduce(mge_opr, builder):
    ReducerOptions.ReducerOptionsStart(builder)
    ReducerOptions.ReducerOptionsAddKeepDims(builder, True)
    options = ReducerOptions.ReducerOptionsEnd(builder)

    reduce_mode_map = {
        "SUM": BuiltinOperator.SUM,
        "MEAN": BuiltinOperator.MEAN,
        "MAX": BuiltinOperator.REDUCE_MAX,
        "MIN": BuiltinOperator.REDUCE_MIN,
    }
    return reduce_mode_map[mge_opr.mode], BuiltinOptions.ReducerOptions, options


@_register_op(ReshapeOpr, AxisAddRemoveOpr)
def _reshape(mge_opr, builder):
    ReshapeOptions.ReshapeOptionsStartNewShapeVector(builder, len(mge_opr.output_shape))
    for i in reversed(list(mge_opr.output_shape)):
        builder.PrependInt32(i)
    new_shape = builder.EndVector(len(mge_opr.output_shape))
    ReshapeOptions.ReshapeOptionsStart(builder)
    ReshapeOptions.ReshapeOptionsAddNewShape(builder, new_shape)
    options = ReshapeOptions.ReshapeOptionsEnd(builder)
    return BuiltinOperator.RESHAPE, BuiltinOptions.ReshapeOptions, options


@_register_op(ConcatOpr)
def _concat(mge_opr, builder):
    ConcatenationOptions.ConcatenationOptionsStart(builder)
    ConcatenationOptions.ConcatenationOptionsAddFusedActivationFunction(
        builder, mge2tflite_activation_type[mge_opr.activation]
    )
    axis = mge_opr.axis
    if mge_opr.inp_vars[0].ndim == 4:
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


@_register_op(PoolingForwardOpr)
def _pooling(mge_opr, builder):
    Pool2DOptions.Pool2DOptionsStart(builder)
    Pool2DOptions.Pool2DOptionsAddPadding(builder, Padding.VALID)
    shape = mge_opr.inp_vars[0].shape
    if get_platform() == "mtk" and shape[2] == mge_opr.kh and shape[3] == mge_opr.kw:
        # MTK global pooling
        print(
            "\nWARNING: the stride of global pooling "
            "would be changed to 1 to adapted the bug of MTK"
        )
        Pool2DOptions.Pool2DOptionsAddStrideH(builder, 1)
        Pool2DOptions.Pool2DOptionsAddStrideW(builder, 1)
    else:
        Pool2DOptions.Pool2DOptionsAddStrideH(builder, mge_opr.sh)
        Pool2DOptions.Pool2DOptionsAddStrideW(builder, mge_opr.sw)
    Pool2DOptions.Pool2DOptionsAddFilterHeight(builder, mge_opr.kh)
    Pool2DOptions.Pool2DOptionsAddFilterWidth(builder, mge_opr.kw)
    Pool2DOptions.Pool2DOptionsAddFusedActivationFunction(
        builder, mge2tflite_activation_type[mge_opr.activation]
    )
    options = Pool2DOptions.Pool2DOptionsEnd(builder)

    tfl_opr_type = BuiltinOperator.AVERAGE_POOL_2D
    if mge_opr.mode == "MAX":
        tfl_opr_type = BuiltinOperator.MAX_POOL_2D
    return tfl_opr_type, BuiltinOptions.Pool2DOptions, options


@_register_op(ConvolutionForwardOpr, ConvBiasForwardOpr)
def _conv2d(mge_opr, builder):
    if mge_opr.group > 1:
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsStart(builder)
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPadding(builder, Padding.VALID)
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideH(builder, mge_opr.sh)
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideW(builder, mge_opr.sw)
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDepthMultiplier(
            builder, mge_opr.inp_vars[1].shape[0]
        )
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddFusedActivationFunction(
            builder, mge2tflite_activation_type[mge_opr.activation]
        )
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationHFactor(
            builder, mge_opr.dilation_h
        )
        DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationWFactor(
            builder, mge_opr.dilation_w
        )
        options = DepthwiseConv2DOptions.DepthwiseConv2DOptionsEnd(builder)
        return (
            BuiltinOperator.DEPTHWISE_CONV_2D,
            BuiltinOptions.DepthwiseConv2DOptions,
            options,
        )

    Conv2DOptions.Conv2DOptionsStart(builder)
    Conv2DOptions.Conv2DOptionsAddPadding(builder, Padding.VALID)
    Conv2DOptions.Conv2DOptionsAddStrideH(builder, mge_opr.sh)
    Conv2DOptions.Conv2DOptionsAddStrideW(builder, mge_opr.sw)
    Conv2DOptions.Conv2DOptionsAddDilationHFactor(builder, mge_opr.dilation_h)
    Conv2DOptions.Conv2DOptionsAddDilationWFactor(builder, mge_opr.dilation_w)
    Conv2DOptions.Conv2DOptionsAddFusedActivationFunction(
        builder, mge2tflite_activation_type[mge_opr.activation]
    )
    options = Conv2DOptions.Conv2DOptionsEnd(builder)
    return BuiltinOperator.CONV_2D, BuiltinOptions.Conv2DOptions, options


@_register_op(ResizeForwardOpr)
def _resize(_, builder):
    ResizeBilinearOptions.ResizeBilinearOptionsStart(builder)
    ResizeBilinearOptions.ResizeBilinearOptionsAddAlignCorners(builder, False)
    options = ResizeBilinearOptions.ResizeBilinearOptionsEnd(builder)
    return (
        BuiltinOperator.RESIZE_BILINEAR,
        BuiltinOptions.ResizeBilinearOptions,
        options,
    )


@_register_op(MatrixMulOpr)
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


@_register_op(SoftmaxOpr)
def _softmax(mge_opr, builder):
    SoftmaxOptions.SoftmaxOptionsStart(builder)
    SoftmaxOptions.SoftmaxOptionsAddBeta(builder, mge_opr.beta)
    options = SoftmaxOptions.SoftmaxOptionsEnd(builder)
    return BuiltinOperator.SOFTMAX, BuiltinOptions.SoftmaxOptions, options


def _padding_mode_transpose_conv(mge_opr):
    if (
        mge_opr.out_vars[0].shape[2] == mge_opr.inp_vars[1].shape[2] * mge_opr.sh
        and mge_opr.out_vars[0].shape[3] == mge_opr.inp_vars[1].shape[3] * mge_opr.sw
    ):
        # padding mode == SAME
        return Padding.SAME
    elif mge_opr.ph == 0 and mge_opr.pw == 0:
        # padding mode == VALID
        return Padding.VALID
    else:
        assert False, "ERROR: unsupported padding mode"
        return None


@_register_op(ConvolutionBackwardDataOpr)
def _deconv(mge_opr, builder):
    if get_platform() == "mtk":
        CustomOperator = collections.namedtuple("CustomOperator", ["code"])
        CustomOptions = collections.namedtuple("CustomOptions", ["code"])

        options = dict()
        options["PaddingType"] = _padding_mode_transpose_conv(mge_opr)
        options["stride_height"] = mge_opr.sh
        options["stride_width"] = mge_opr.sw
        options["depth_multiplier"] = 1
        options["dilation_height_factor"] = mge_opr.dilation_h
        options["dilation_width_factor"] = mge_opr.dilation_w
        options["activation"] = mge2tflite_activation_type[mge_opr.activation]
        return (
            CustomOperator("MTK_TRANSPOSE_CONV"),
            CustomOptions("MTK_TRANSPOSE_CONV"),
            dumps(options),
        )

    TransposeConvOptions.TransposeConvOptionsStart(builder)
    TransposeConvOptions.TransposeConvOptionsAddPadding(
        builder, _padding_mode_transpose_conv(mge_opr)
    )
    TransposeConvOptions.TransposeConvOptionsAddStrideH(builder, mge_opr.sh)
    TransposeConvOptions.TransposeConvOptionsAddStrideW(builder, mge_opr.sw)
    options = TransposeConvOptions.TransposeConvOptionsEnd(builder)
    return BuiltinOperator.TRANSPOSE_CONV, BuiltinOptions.TransposeConvOptions, options


@_register_op(PadOpr)
def _pad(_, builder):
    PadOptions.PadOptionsStart(builder)
    options = PadOptions.PadOptionsEnd(builder)
    return BuiltinOperator.PAD, BuiltinOptions.PadOptions, options


@_register_op(DimshuffleOpr)
def _dimshuffle(_, builder):
    TransposeOptions.TransposeOptionsStart(builder)
    options = TransposeOptions.TransposeOptionsEnd(builder)
    return BuiltinOperator.TRANSPOSE, BuiltinOptions.TransposeOptions, options
