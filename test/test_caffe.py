# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import caffe  # pylint: disable=import-error
import megengine
import megengine.hub
import numpy as np
import pytest
from mgeconvert.caffe_converter import convert_to_caffe

from .utils import (
    ActiveOpr,
    BnOpr,
    BroadcastOpr,
    ConcatOpr,
    ConvOpr,
    ElemwiseOpr,
    LinearOpr,
    PoolOpr,
    ReduceOpr,
    ReshapeOpr,
    SoftmaxOpr,
    SqueezeOpr,
    SubtensorOpr,
    TransposeOpr,
    XORNet,
    dump_mge_model,
)

max_error = 1e-6
tmp_file = "test_model"


def _test_convert_result(inputs, fpath, mge_results, max_err):
    convert_to_caffe(
        fpath + ".mge", prototxt=tmp_file + ".txt", caffemodel=tmp_file + ".caffemodel"
    )
    caffe_net = caffe.Net(tmp_file + ".txt", "test_model.caffemodel", caffe.TEST)
    for i in caffe_net.blobs.keys():
        if "data" in i:
            caffe_net.blobs[i].data[...] = inputs
            break
    caffe_net.forward()
    caffe_dict = caffe_net.blobs
    caffe_results = list(caffe_dict.items())[-1][1].data
    assert caffe_results.shape == mge_results.shape
    assert np.allclose(caffe_results, mge_results, atol=max_err)


@pytest.mark.parametrize("mode", ["normal", "group", "transpose"])
def test_conv2d(mode):
    net = ConvOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_linear():
    net = LinearOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_softmax():
    net = SoftmaxOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_squeeze():
    net = SqueezeOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["max", "avg"])
def test_pooling(mode):
    if megengine.__version__ > "0.6.0" and mode == "avg":
        return
    net = PoolOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["bn1d", "bn2d"])
def test_batchnorm(mode):
    net = BnOpr(mode)
    data = net.data1 if mode == "bn1d" else net.data2
    mge_result = dump_mge_model(net, data, tmp_file)
    _test_convert_result(data, tmp_file, mge_result, max_error)


def test_subtensor():
    net = SubtensorOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_transopse():
    net = TransposeOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_concat():
    net = ConcatOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_reshape():
    net = ReshapeOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize(
    "mode", ["add", "sub", "mul", "div", "abs", "exp", "log", "max", "pow"]
)
def test_elemwise(mode):
    net = ElemwiseOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize(
    "mode", ["add", "sub", "mul", "div", "abs", "exp", "log", "pow"]
)
def test_elemwise_broadcast(mode):
    net = ElemwiseOpr(mode)
    mge_result = dump_mge_model(net, np.array([2.0]).astype("float32"), tmp_file)
    _test_convert_result(np.array([2.0]), tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["relu", "sigmoid", "tanh", "leaky_relu"])
def test_active(mode):
    net = ActiveOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["max", "sum", "mean"])
def test_reduce(mode):
    net = ReduceOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_broadcast():
    net = BroadcastOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


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
    mge_result = dump_mge_model(net, data, tmp_file)
    _test_convert_result(data, tmp_file, mge_result, 1e-2)


def test_xornet():
    if megengine.__version__ < "1.1.0":
        return
    net = XORNet()
    mge_result = dump_mge_model(net, net.data, tmp_file, True)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_leakyrelu_model():
    if megengine.__version__ < "1.1.0":
        return
    net = XORNet()
    mge_result = dump_mge_model(net, net.data, tmp_file, False)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)
