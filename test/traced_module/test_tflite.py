# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint:disable=import-outside-toplevel, no-name-in-module,import-error
from test.traced_module.tm_utils import get_traced_module
from test.utils import (
    ConvBn2dOpr,
    ConvBnRelu2dOpr,
    ConvOpr,
    ConvRelu2dOpr,
    ElemwiseOpr,
    FConcatOpr,
    LinearOpr,
    NCHW_SubtensorOpr,
    PadOpr,
    PoolOpr,
    ReduceOpr,
    ReshapeOpr,
    ResizeOpr,
    SoftmaxOpr,
    SqueezeOpr,
    SubtensorOpr,
    TransposeOpr,
    TypeCvtOpr,
)
from typing import Sequence

import megengine as mge
import megengine.functional as F
import megengine.hub
import megengine.module as M
import numpy as np
import pytest
from megengine.traced_module import trace_module
from mgeconvert.converters.tm_to_tflite import tracedmodule_to_tflite
from tensorflow.lite.python import interpreter  # pylint: disable=import-error

max_error = 1e-6
tmp_file = "test_model"


def _test_convert_result(
    inputs,
    tm,
    tm_result,
    max_err=max_error,
    nhwc=True,
    nhwc2=True,
    scale=1,
    zero_point=0,
    require_quantize=False,
):
    if not isinstance(inputs, Sequence):
        inputs = [
            inputs,
        ]
    if not isinstance(scale, Sequence):
        scale = (scale,)
    if not isinstance(zero_point, Sequence):
        zero_point = (zero_point,)
    for i, inp in enumerate(inputs):
        if nhwc and inp.ndim == 4:
            inputs[i] = inp.transpose((0, 2, 3, 1))

    tracedmodule_to_tflite(
        tm,
        output=tmp_file + ".tflite",
        require_quantize=require_quantize,
        disable_nhwc=not nhwc,
    )

    tfl_model = interpreter.Interpreter(model_path=tmp_file + ".tflite")
    tfl_model.allocate_tensors()

    input_details = tfl_model.get_input_details()
    for i, inp in enumerate(inputs):
        tfl_model.set_tensor(input_details[i]["index"], inp)
    tfl_model.invoke()

    pred_tfl = []
    if not isinstance(scale, Sequence):
        scale = (scale,)
        zero_point = (zero_point,)
    for index, i in enumerate(tfl_model.get_output_details()):
        out = tfl_model.tensor(i["index"])()
        if nhwc2 and out.ndim == 4:
            out = out.transpose((0, 3, 1, 2))
        index = len(scale) - 1 if index >= len(scale) else index
        out = ((out - float(zero_point[index])) * scale[index]).astype("float32")
        pred_tfl.append(out)

    if not isinstance(tm_result, Sequence):
        tm_result = (tm_result,)
    for i, j, s in zip(tm_result, pred_tfl, scale):
        assert i.shape == j.shape
        assert i.dtype == j.dtype
        atol = max_err if s == 1 else s
        np.testing.assert_allclose(i, j, atol=atol)


@pytest.mark.parametrize(
    "mode",
    [
        "normal",
        "group",
        "tflite_transpose",
        "same_pad",
        "same_pad_1",
        "same_pad_2",
        "valid_pad",
        "valid_pad_1",
    ],
)
def test_conv(mode):
    net = ConvOpr(mode)
    data = mge.tensor(np.random.random((1, 3, 224, 224)).astype(np.float32))
    traced_module, tm_result = get_traced_module(net, data)
    print(traced_module.flatten().graph)

    _test_convert_result(data, traced_module, tm_result)


def test_convrelu():
    net = ConvRelu2dOpr()
    data = mge.tensor(net.data)
    traced_module, tm_result = get_traced_module(net, data)
    print(traced_module.flatten().graph)

    _test_convert_result(data, traced_module, tm_result)


@pytest.mark.parametrize(
    "has_bias", [True, False],
)
def test_convbn(has_bias):
    net = ConvBn2dOpr(has_bias)
    net.eval()
    data = mge.tensor(net.data)
    traced_module, tm_result = get_traced_module(net, data)
    print(traced_module.flatten().graph)

    _test_convert_result(data, traced_module, tm_result)


@pytest.mark.parametrize(
    "has_bias", [True, False],
)
def test_convbnrelu(has_bias):
    net = ConvBnRelu2dOpr(has_bias)
    net.eval()
    data = mge.tensor(net.data)
    traced_module, tm_result = get_traced_module(net, data)
    print(traced_module.flatten().graph)

    _test_convert_result(data, traced_module, tm_result)


def test_reshape():
    net = ReshapeOpr(fix_batch=True)
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    print(traced_module.flatten().graph)
    _test_convert_result(mge.tensor(net.data), traced_module, tm_result, nhwc=False)


@pytest.mark.parametrize("mode", ["max", "min", "sum", "mean"])
def test_reduce(mode):
    net = ReduceOpr(mode=mode)
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    print(traced_module.flatten().graph)
    _test_convert_result(mge.tensor(net.data), traced_module, tm_result)


@pytest.mark.parametrize(
    "mode",
    [
        "pow",
        "exp",
        "min",
        "max",
        "add",
        "div",
        "sub",
        "mul",
        "fuse_add_relu",
        "fuse_add_sigmoid",
    ],
)
def test_elemwise(mode):
    net = ElemwiseOpr(mode)
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    print(traced_module.flatten().graph)

    _test_convert_result(mge.tensor(net.data), traced_module, tm_result)


@pytest.mark.parametrize("mode", ["max", "avg"])
def test_pooling(mode):
    if mge.__version__ > "0.6.0" and mode == "avg":
        return
    net = PoolOpr(mode)
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    print(traced_module.flatten().graph)

    _test_convert_result(mge.tensor(net.data), traced_module, tm_result)


def test_pad():
    net = PadOpr()
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(mge.tensor(net.data), traced_module, tm_result)


def test_linear():
    net = LinearOpr()
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    print(traced_module.flatten().graph)
    _test_convert_result(mge.tensor(net.data), traced_module, tm_result)


def test_transopse():
    net = TransposeOpr()
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    print(traced_module.flatten().graph)
    _test_convert_result(mge.tensor(net.data), traced_module, tm_result)


def test_softmax():
    net = SoftmaxOpr()
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    print(traced_module.flatten().graph)
    _test_convert_result(mge.tensor(net.data), traced_module, tm_result)


def test_squeeze():
    net = SqueezeOpr()
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    print(traced_module.flatten().graph)
    _test_convert_result(
        mge.tensor(net.data), traced_module, tm_result, nhwc=False, nhwc2=False
    )


def test_slice():
    net = SubtensorOpr()
    tm, tm_result = get_traced_module(net, mge.tensor(net.data))
    print(tm.flatten().graph)
    _test_convert_result(mge.tensor(net.data), tm, tm_result, nhwc=False, nhwc2=False)
    net1 = NCHW_SubtensorOpr()
    tm, tm_result = get_traced_module(net1, mge.tensor(net1.data))
    tm_result = mge.tensor(net1.data).transpose(0, 2, 3, 1)[1:3, 4:9, 2, 4:8]
    _test_convert_result(mge.tensor(net1.data), tm, tm_result, nhwc=True, nhwc2=False)


def test_typecvt():
    net = TypeCvtOpr()
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    print(traced_module.flatten().graph)
    _test_convert_result(mge.tensor(net.data), traced_module, tm_result)


def test_resize():
    net = ResizeOpr()
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    print(traced_module.flatten().graph)
    _test_convert_result(mge.tensor(net.data), traced_module, tm_result)


def test_F_concat():
    net = FConcatOpr()
    data1 = mge.tensor(np.random.random((1, 3, 24, 24)).astype(np.float32))
    data2 = mge.tensor(np.random.random((1, 3, 24, 24)).astype(np.float32))
    traced_module, tm_result = get_traced_module(net, [data1, data2])
    print(traced_module.flatten().graph)
    _test_convert_result([data1, data2], traced_module, tm_result)


def test_float_func_conv():
    class FConvOpr(M.Module):
        def __init__(self):
            super().__init__()
            self.conv = F.conv2d

        def forward(
            self,
            inp,
            weight,
            bias=None,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
        ):
            x = F.conv2d(inp, weight, bias, stride, padding, dilation, groups)
            return x

    net = FConvOpr()
    data = mge.tensor(np.random.random((1, 16, 32, 32))).astype("float32")
    weight = mge.tensor(np.random.random((32, 16, 2, 2))).astype("float32")
    traced_module = trace_module(net, data, weight)
    tm_result = traced_module(data, weight)
    _test_convert_result([data, weight], traced_module, tm_result, max_err=1e-4)


@pytest.mark.parametrize(
    "model", ["resnet18",],
)
def test_model(model):
    data = np.ones((1, 3, 224, 224)).astype(np.float32)
    if megengine.__version__ < "1.1.0":
        commit_id = "dc2f2cfb228a135747d083517b98aea56e7aab92"
    else:
        commit_id = None
    net = megengine.hub.load(
        "megengine/models", model, use_cache=False, commit=commit_id, pretrained=True
    )
    net.eval()
    traced_module, tm_result = get_traced_module(net, mge.tensor(data))
    _test_convert_result(mge.tensor(data), traced_module, tm_result, 1e-4)
