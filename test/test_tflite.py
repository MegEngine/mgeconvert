# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine
import numpy as np
import pytest
from mgeconvert.mge_context import TopologyNetwork
from mgeconvert.tflite_converter.tflite_converter import TFLiteConverter
from tensorflow.lite.python import interpreter

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


def _test_convert_result(inputs, fpath, mge_result, max_err, nhwc=True):
    if nhwc and inputs.ndim == 4:
        inputs = inputs.transpose((0, 2, 3, 1))
    print("-- mge result shape\n", mge_result)
    net = TopologyNetwork(fpath + ".mge")

    converter = TFLiteConverter(net, graph_name="graph")
    model = converter.convert()
    with open(tmp_file + ".tflite", "wb") as fout:
        fout.write(model)

    tfl_model = interpreter.Interpreter(model_path=tmp_file + ".tflite")
    tfl_model.allocate_tensors()

    input_details = tfl_model.get_input_details()
    tfl_model.set_tensor(input_details[0]["index"], inputs)
    tfl_model.invoke()
    pred_tfl = tfl_model.tensor(tfl_model.get_output_details()[0]["index"])()
    if pred_tfl.ndim == 4:
        pred_tfl = pred_tfl.transpose((0, 3, 1, 2))
    print("@@ predict tflite shape\n", pred_tfl)
    assert pred_tfl.shape == mge_result.shape
    assert pred_tfl.dtype == mge_result.dtype
    assert np.allclose(pred_tfl, mge_result, atol=max_err)


def test_conv2d():
    net = ConvOpr("normal")
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


@pytest.mark.parametrize("mode", ["max", "avg"])
def test_pooling(mode):
    if megengine.__version__ > "0.6.0" and mode == "avg":
        return
    net = PoolOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_concat():
    net = ConcatOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_reshape():
    net = ReshapeOpr(fix_batch=True)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error, nhwc=False)


@pytest.mark.parametrize("mode", ["add", "sub", "mul", "div", "exp", "max"])
def test_elemwise(mode):
    net = ElemwiseOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["add", "sub", "mul", "div", "exp", "max"])
def test_elemwise_broadcast(mode):
    net = ElemwiseOpr(mode)
    mge_result = dump_mge_model(net, np.array([2.0]).astype("float32"), tmp_file)
    _test_convert_result(
        np.array([2.0]).astype("float32"), tmp_file, mge_result, max_error
    )


@pytest.mark.parametrize("mode", ["max", "sum"])
def test_reduce(mode):
    net = ReduceOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["relu", "softmax", "relu6"])
def test_active(mode):
    net = ActiveOpr(mode, fused=True)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)
