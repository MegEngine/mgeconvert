# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import sys

import megengine as mge
import megengine.hub
import numpy as np
import pytest
import torch
from mgeconvert.converters.tm_to_torchscript import tracedmodule_to_torchscript

from .tm_utils import get_traced_module

from test.utils import (  # AdaptiveAvgPool2dOpr,; DropoutOpr,; ElemwiseOpr,; FlattenOpr,; PoolOpr,; RepeatOpr,
    ActiveOpr,
    BnOpr,
    BroadcastOpr,
    ConvBn2dOpr,
    ConvBnRelu2dOpr,
    ConvOpr,
    ConvRelu2dOpr,
    ElemwiseOpr,
    FConcatOpr,
    FlattenOpr,
    LinearOpr,
    ReduceOpr,
    RepeatOpr,
    ReshapeOpr,
    ResizeOpr,
    SqueezeOpr,
    SubtensorOpr,
    TransposeOpr,
    TypeCvtOpr,
)


max_error = 1e-6
tmp_file = "test_model"


def _test_convert_result(inputs, fpath, mge_result, max_err):
    tracedmodule_to_torchscript(fpath, tmp_file + ".torchscript", graph_name="graph")
    if isinstance(inputs, np.ndarray):
        inputs = [inputs]
    inputs = tuple([torch.from_numpy(inp) for inp in inputs])
    script_model = torch.load(tmp_file + ".torchscript",)
    script_model.eval()
    pred_torch = script_model(*inputs)
    pred_torch = pred_torch.numpy()
    assert pred_torch.shape == mge_result.shape
    assert pred_torch.dtype == mge_result.dtype
    np.testing.assert_allclose(pred_torch, mge_result, rtol=max_err, atol=max_err)


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


# @pytest.mark.parametrize("mode", ["max", "avg"])
# def test_pool(mode):
#     if mode == "avg":
#         return
#     net = PoolOpr(mode)
#     tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
#     _test_convert_result(net.data, tm_module, mge_result, max_error)


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
    _test_convert_result(net.data, tm_module, mge_result, max_error)


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
    "mode",
    [
        "relu",
        "relu6",
        # "tanh",
        "sigmoid",
        # "hswish", "hsigmoid", "silu"
    ],
)
def test_active(mode):
    if megengine.__version__ < "1.5.0" and mode == "silu":
        return
    net = ActiveOpr(mode)
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


# def test_broadcast():
#     net = BroadcastOpr()
#     tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
#     _test_convert_result(net.data, tm_module, mge_result, max_error, min_version=8)


def test_typecvt():
    net = TypeCvtOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_flatten():
    net = FlattenOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


# def test_dropout():
#     net = DropoutOpr()
#     tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
#     _test_convert_result(net.data, tm_module, mge_result, max_error)


# def test_globalpooling():
#     net = AdaptiveAvgPool2dOpr()
#     tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
#     _test_convert_result(net.data, tm_module, mge_result, max_error)


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
