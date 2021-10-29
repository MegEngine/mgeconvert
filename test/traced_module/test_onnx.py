# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from test.utils import (
    ActiveOpr,
    AdaptiveAvgPool2dOpr,
    BnOpr,
    BroadcastOpr,
    ConvBn2dOpr,
    ConvBnRelu2dOpr,
    ConvOpr,
    ConvRelu2dOpr,
    DropoutOpr,
    ElemwiseOpr,
    FConcatOpr,
    FlattenOpr,
    LinearOpr,
    PoolOpr,
    ReduceOpr,
    RepeatOpr,
    ReshapeOpr,
    ResizeOpr,
    SqueezeOpr,
    SubtensorOpr,
    TransposeOpr,
    TypeCvtOpr,
)

import megengine as mge
import megengine.hub
import numpy as np
import onnxruntime as ort
import pytest
from mgeconvert.converters.tm_to_onnx import tracedmodule_to_onnx

from .tm_utils import get_traced_module

max_error = 1e-6
tmp_file = "test_model"


def _test_convert_result(
    inputs, fpath, mge_result, max_err, min_version=7, max_version=12
):
    for version in range(min_version, max_version + 1):
        tracedmodule_to_onnx(
            fpath, tmp_file + ".onnx", opset=version, graph_name="graph"
        )
        onnx_net = ort.InferenceSession(tmp_file + ".onnx")
        if isinstance(inputs, (list, tuple)):
            input_dict = {}
            for i, inp in enumerate(inputs):
                input_name = onnx_net.get_inputs()[i].name
                X_test = inp
                input_dict[input_name] = X_test
            pred_onx = onnx_net.run(None, input_dict)[0]
        else:
            input_name = onnx_net.get_inputs()[0].name
            X_test = inputs
            pred_onx = onnx_net.run(None, {input_name: X_test})[0]
        assert pred_onx.shape == mge_result.shape
        assert pred_onx.dtype == mge_result.dtype
        np.testing.assert_allclose(pred_onx, mge_result, rtol=max_err, atol=max_err)


@pytest.mark.parametrize("mode", ["normal", "group", "transpose"])
def test_conv2d(mode):
    net = ConvOpr(mode)
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_convrelu():
    net = ConvRelu2dOpr()
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, traced_module, tm_result, max_error)


def test_convbn():
    net = ConvBn2dOpr()
    net.eval()
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, traced_module, tm_result, max_error)


def test_convbnrelu():
    net = ConvBnRelu2dOpr()
    net.eval()
    traced_module, tm_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, traced_module, tm_result, max_error)


def test_linear():
    net = LinearOpr()
    net.eval()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


@pytest.mark.parametrize("mode", ["max", "avg"])
def test_pool(mode):
    if mode == "avg":
        return
    net = PoolOpr(mode)
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


@pytest.mark.parametrize("mode", ["bn1d", "bn2d"])
def test_batchnorm(mode):
    net = BnOpr(mode)
    net.eval()
    data = net.data1 if mode == "bn1d" else net.data2
    tm_module, mge_result = get_traced_module(net, mge.tensor(data))
    _test_convert_result(data, tm_module, mge_result, max_error)


def test_subtensor():
    net = SubtensorOpr()
    net.eval()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_transpose():
    net = TransposeOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_concat():
    net = FConcatOpr()
    data = np.random.random((1, 2, 4, 5)).astype(np.float32)
    list_data = [mge.tensor(data), mge.tensor(data)]
    tm_module, mge_result = get_traced_module(net, list_data)
    _test_convert_result([data, data], tm_module, mge_result, max_error)


def test_softmax():
    net = ActiveOpr(mode="softmax")
    data = np.random.random((10, 8)).astype(np.float32)
    tm_module, mge_result = get_traced_module(net, mge.tensor(data))
    _test_convert_result(data, tm_module, mge_result, max_error)


def test_squeeze():
    net = SqueezeOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_reshape():
    net = ReshapeOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_resize():
    net = ResizeOpr(mode="nearest")
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(
        net.data, tm_module, mge_result, max_error, min_version=10, max_version=10
    )


@pytest.mark.parametrize(
    "mode",
    ["add", "sub", "mul", "div", "abs", "exp", "log", "pow", "ceil", "floor", "max",],
)
def test_elemwise(mode):
    net = ElemwiseOpr(mode)
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


@pytest.mark.parametrize("mode", ["sum", "max"])
def test_reduce(mode):
    net = ReduceOpr(mode)
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


@pytest.mark.parametrize(
    "mode", ["relu", "relu6", "tanh", "sigmoid", "hswish", "hsigmoid", "silu"]
)
def test_active(mode):
    if megengine.__version__ < "1.5.0" and mode == "silu":
        return
    net = ActiveOpr(mode)
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_broadcast():
    net = BroadcastOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error, min_version=8)


def test_typecvt():
    net = TypeCvtOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_flatten():
    net = FlattenOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_dropout():
    net = DropoutOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_globalpooling():
    net = AdaptiveAvgPool2dOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_repeat():
    net = RepeatOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


@pytest.mark.parametrize(
    "model",
    [
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        "resnet18",
        "resnet50",
        "resnet101",
        "resnext50_32x4d",
    ],
)
def test_model(model):
    data = (
        np.random.randint(0, 255, 3 * 224 * 224)
        .reshape((1, 3, 224, 224))
        .astype(np.float32)
    )
    if mge.__version__ < "1.1.0":
        commit_id = "dc2f2cfb228a135747d083517b98aea56e7aab92"
    else:
        commit_id = None
    net = megengine.hub.load(
        "megengine/models", model, use_cache=False, commit=commit_id, pretrained=True
    )
    net.eval()
    tm_module, mge_result = get_traced_module(net, mge.tensor(data))
    _test_convert_result(data, tm_module, mge_result, 1e-2)
