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
    LinearBnOpr,
    LinearOpr,
    MatrixMulBnOpr,
    PoolOpr,
    ReduceOpr,
    RepeatOpr,
    ReshapeOpr,
    SqueezeOpr,
    SubtensorOpr,
    TransposeOpr,
    XORNet,
    XORNet_LeakyRelu,
)

import caffe  # pylint: disable=import-error
import megengine as mge
import megengine.hub
import numpy as np
import pytest
from mgeconvert.converters.tm_to_caffe import tracedmodule_to_caffe

from .tm_utils import get_traced_module

max_error = 1e-6
tmp_file = "test_module"


def _test_convert_result(
    inputs,
    trace_module,
    mge_results,
    max_err,
    input_data_type=None,
    input_scales=None,
    input_zero_points=None,
    require_quantize=False,
    param_fake_quant=False,
    split_conv_relu=False,
    fuse_bn=False,
    input_name="x",
    convert_backend=1,
):
    tracedmodule_to_caffe(
        trace_module,
        prototxt=tmp_file + ".txt",
        caffemodel=tmp_file + ".caffemodel",
        input_data_type=input_data_type,
        input_scales=input_scales,
        input_zero_points=input_zero_points,
        require_quantize=require_quantize,
        param_fake_quant=param_fake_quant,
        split_conv_relu=split_conv_relu,
        fuse_bn=fuse_bn,
        convert_backend=convert_backend,
    )

    caffe_net = caffe.Net(tmp_file + ".txt", tmp_file + ".caffemodel", caffe.TEST)
    for i in caffe_net.blobs.keys():
        if isinstance(input_name, list):
            for idx, name in enumerate(input_name):
                if name.strip() == i.strip():
                    caffe_net.blobs[i].data[...] = inputs[idx]
                    break
        else:
            if input_name in i:
                caffe_net.blobs[i].data[...] = inputs
                break
    out_dict = caffe_net.forward()

    if isinstance(mge_results, dict):
        assert len(list(out_dict.keys())) == len(list(mge_results.keys()))
        for name in mge_results.keys():
            assert name._name in out_dict.keys()
            assert out_dict[name._name].shape == mge_results[name].shape
            np.testing.assert_allclose(
                out_dict[name._name], mge_results[name], atol=max_err
            )
    else:
        caffe_results = list(out_dict.values())[0]
        assert caffe_results.shape == mge_results.shape
        np.testing.assert_allclose(
            caffe_results, mge_results, rtol=max_err, atol=max_err
        )


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
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_flatten_linear():
    net = LinearOpr("flatten")
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data1))
    _test_convert_result(net.data1, tm_module, mge_result, max_error, convert_backend=4)


def test_linear_bn():
    net = LinearBnOpr()
    for _ in range(10):
        net(mge.tensor(net.data)).numpy()
    net.eval()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, 1e-4, fuse_bn=True)


@pytest.mark.parametrize("mode", [True, False])
def test_matmul_bn(mode):
    net = MatrixMulBnOpr(mode)
    for _ in range(10):
        net(mge.tensor(net.data)).numpy()
    net.eval()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, 1e-4, fuse_bn=True)


def test_squeeze():
    net = SqueezeOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error, input_name="a")


@pytest.mark.parametrize("mode", ["max", "avg"])
def test_pooling(mode):
    if megengine.__version__ > "0.6.0" and mode == "avg":
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
    _test_convert_result(
        [data, data], tm_module, mge_result, max_error, input_name=["inps_0", "inps_1"]
    )


def test_reshape():
    net = ReshapeOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


@pytest.mark.parametrize(
    "mode", ["add", "sub", "mul", "div", "abs", "exp", "log", "max", "pow"]
)
def test_elemwise(mode):
    net = ElemwiseOpr(mode)
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error, input_name="a")


@pytest.mark.parametrize(
    "mode", ["add", "sub", "mul", "div", "abs", "exp", "log", "pow"]
)
def test_elemwise_broadcast(mode):
    net = ElemwiseOpr(mode)
    tm_module, mge_result = get_traced_module(
        net, mge.tensor(np.array([2.0]).astype("float32"))
    )
    _test_convert_result(
        np.array([2.0]), tm_module, mge_result, max_error, input_name="a"
    )


@pytest.mark.parametrize(
    "mode",
    [
        "relu",
        "sigmoid",
        "tanh",
        "leaky_relu",
        "softmax",
        "silu",
        "relu6",
        "hsigmoid",
        "hswish",
    ],
)
def test_active(mode):
    if megengine.__version__ < "1.5.0" and mode == "silu":
        return
    net = ActiveOpr(mode)
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


@pytest.mark.parametrize("mode", ["relu",])
def test_active_inplace(mode):
    net = ActiveOpr(mode)
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error, convert_backend=4)


@pytest.mark.parametrize("mode", ["max", "sum", "mean"])
def test_reduce(mode):
    net = ReduceOpr(mode)
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error, input_name="a")


def test_broadcast():
    net = BroadcastOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_repeat():
    net = RepeatOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_flatten():
    net = FlattenOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error, input_name="inps")


def test_dropout():
    net = DropoutOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error, input_name="inps")


def test_adapetive_avg_pool():
    net = AdaptiveAvgPool2dOpr()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error, input_name="inps")


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
    if megengine.__version__ < "1.1.0":
        commit_id = "dc2f2cfb228a135747d083517b98aea56e7aab92"
    else:
        commit_id = None
    net = megengine.hub.load(
        "megengine/models", model, use_cache=False, commit=commit_id, pretrained=True
    )
    net.eval()
    tm_module, mge_result = get_traced_module(net, mge.tensor(data))
    _test_convert_result(data, tm_module, mge_result, 1e-2)


def test_xornet():
    if megengine.__version__ < "1.1.0":
        return
    net = XORNet()
    net.eval()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)


def test_leakyrelu_model():
    if megengine.__version__ < "1.1.0":
        return
    net = XORNet_LeakyRelu()
    net.eval()
    tm_module, mge_result = get_traced_module(net, mge.tensor(net.data))
    _test_convert_result(net.data, tm_module, mge_result, max_error)
