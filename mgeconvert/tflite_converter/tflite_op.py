# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from ..mge_context import (
    AxisAddRemoveOpr,
    BatchNormForwardOpr,
    BroadcastOpr,
    ConcatOpr,
    ConvolutionBackwardDataOpr,
    ConvolutionForwardOpr,
    DimshuffleOpr,
    ElemwiseOpr,
    GetVarShapeOpr,
    Host2DeviceCopyOpr,
    IdentityOpr,
    MarkNoBroadcastElemwiseOpr,
    MatrixMulOpr,
    MultipleDeviceTensorHolderOpr,
    PoolingForwardOpr,
    ReduceOpr,
    ReshapeOpr,
    SharedDeviceTensorOpr,
    SubtensorOpr,
    TypeCvtOpr,
    get_symvar_value,
)
from .tflite import (
    AddOptions,
    ConcatenationOptions,
    Conv2DOptions,
    DepthwiseConv2DOptions,
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
)
from .tflite.BuiltinOperator import BuiltinOperator
from .tflite.BuiltinOptions import BuiltinOptions
from .tflite.TensorType import TensorType

mge2tflite_dtype_mapping = {
    # pylint: disable=no-member
    np.float32: TensorType.FLOAT32,
    np.float16: TensorType.FLOAT16,
    np.int32: TensorType.INT32,
    np.int8: TensorType.INT8,
    np.uint8: TensorType.UINT8,
}


def mge2tflite_opr_type(mge_op):
    # FIXME: add more mapping
    return BuiltinOperator.NEG


def operator2options(tflite_op):
    # FIXME: add more mapping
    return BuiltinOptions.NegOptions


def gen_tflite_options(builder, opr, options_type):
    # FIXME: add more mapping
    NegOptions.NegOptionsStart(builder)
    options = NegOptions.NegOptionsEnd(builder)
    return options
