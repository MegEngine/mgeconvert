# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import random
import sys

import megengine
import megengine.hub
import numpy as np
import onnxruntime as ort
import pytest
from mgeconvert.mge_context import TopologyNetwork
from mgeconvert.onnx_converter import convert_to_onnx
from mgeconvert.onnx_converter.onnx_converter import OnnxConverter

from .utils import *

max_error = 1e-6
onnx_min_version = 7
onnx_max_version = 12
tmp_file = "test_model"


def _test_convert_result(inputs, fpath, mge_result, max_err):
    net = TopologyNetwork(fpath + ".mge")
    for version in range(onnx_min_version, onnx_max_version + 1):
        converter = OnnxConverter(net, opset_version=version, graph_name="graph")
        model = converter.convert()
        with open(tmp_file + ".onnx", "wb") as fout:
            fout.write(model.SerializeToString())
        onnx_net = ort.InferenceSession(tmp_file + ".onnx")
        input_name = onnx_net.get_inputs()[0].name
        X_test = inputs
        pred_onx = onnx_net.run(None, {input_name: X_test.astype(np.float32)})[0]
        assert pred_onx.shape == mge_result.shape
        assert np.allclose(pred_onx, mge_result, atol=max_err)


def test_conv2d():
    for choose in range(2):
        net = ConvOpr(choose)
        mge_result = dump_mge_model(net, net.data, tmp_file)
        _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_linear():
    net = LinearOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["max", "avg"])
def test_pool(mode):
    net = PoolOpr("max")
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
    mge_result = dump_mge_model(net, net.data)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_concat():
    net = ConcatOpr()
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


def test_reshape():
    net = ReshapeOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["add", "sub", "mul", "div", "abs", "exp", "log"])
def test_elemwise(mode):
    net = ElemwiseOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["relu", "tanh", "sigmoid"])
def test_active(mode):
    net = ActiveOpr(mode)
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
    net = megengine.hub.load("megengine/models", model, pretrained=True)
    mge_result = dump_mge_model(net, data, tmp_file)
    _test_convert_result(data, tmp_file, mge_result, 1e-3)


def test_xornet():
    if megengine.__version__ < "1.1.0":
        return
    net = XORNet()
    mge_result = dump_mge_model(net, net.data, tmp_file, True)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)
