# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from test.utils import (
    ActiveOpr,
    BnOpr,
    ConcatOpr,
    ConvBn2dOpr,
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
    TypeCvtOpr,
    XORNet,
    dump_mge_model,
)

import megengine
import megengine.hub
import numpy as np
import pytest
from mgeconvert.converters.mge_to_tflite import mge_to_tflite
from tensorflow.lite.python import interpreter  # pylint: disable=import-error

max_error = 1e-6
tmp_file = "test_model"


def _test_convert_result(
    inputs, fpath, mge_result, max_err, nhwc=True, nhwc2=True, disable_nhwc=False
):
    if nhwc and inputs.ndim == 4:
        inputs = inputs.transpose((0, 2, 3, 1))

    mge_to_tflite(
        fpath + ".mge",
        output=tmp_file + ".tflite",
        mtk=False,
        disable_nhwc=disable_nhwc,
    )

    tfl_model = interpreter.Interpreter(model_path=tmp_file + ".tflite")
    tfl_model.allocate_tensors()

    input_details = tfl_model.get_input_details()
    tfl_model.set_tensor(input_details[0]["index"], inputs)
    tfl_model.invoke()
    pred_tfl = tfl_model.tensor(tfl_model.get_output_details()[0]["index"])()
    if nhwc2 and pred_tfl.ndim == 4:
        pred_tfl = pred_tfl.transpose((0, 3, 1, 2))
    assert pred_tfl.shape == mge_result.shape
    assert pred_tfl.dtype == mge_result.dtype
    np.testing.assert_allclose(pred_tfl, mge_result, atol=max_err)
    print("success!")


@pytest.mark.parametrize("mode", ["normal", "group", "tflite_transpose"])
def test_conv2d(mode):
    net = ConvOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_convbn():
    net = ConvBn2dOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


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
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize(
    "mode", ["fuse_add_relu", "fuse_add_sigmoid",],
)
def test_expand_transform(mode):
    net = ElemwiseOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file, optimize_for_inference=True)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_concat():
    net = ConcatOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_reshape():
    net = ReshapeOpr(fix_batch=True)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error, nhwc=False)


@pytest.mark.parametrize("mode", ["max", "min", "sum", "mean"])
def test_reduce(mode):
    net = ReduceOpr(mode=mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["max", "avg"])
def test_pooling(mode):
    if megengine.__version__ > "0.6.0" and mode == "avg":
        return
    net = PoolOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_linear():
    net = LinearOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_transopse():
    net = TransposeOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_squeeze():
    net = SqueezeOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_slice():
    net = SubtensorOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(
        net.data,
        tmp_file,
        mge_result,
        max_error,
        nhwc=False,
        nhwc2=False,
        disable_nhwc=True,
    )


def test_typecvt():
    net = TypeCvtOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["add", "sub", "mul", "div", "exp", "max", "abs"])
def test_elemwise_broadcast(mode):
    net = ElemwiseOpr(mode)
    mge_result = dump_mge_model(net, np.array([2.0]).astype("float32"), tmp_file)
    _test_convert_result(
        np.array([2.0]).astype("float32"), tmp_file, mge_result, max_error
    )


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
    mge_result = dump_mge_model(net, data, tmp_file)
    _test_convert_result(data, tmp_file, mge_result, 1e-4)


def test_softmax():
    net = SoftmaxOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["relu", "softmax", "relu6", "leaky_relu"])
def test_active(mode):
    net = ActiveOpr(mode, fused=False)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_bn():
    net = BnOpr("bn1d")
    net.eval()
    mge_result = dump_mge_model(net, net.data1, tmp_file, True)
    _test_convert_result(net.data1, tmp_file, mge_result, max_error, disable_nhwc=True)


def test_xornet():
    if megengine.__version__ < "1.1.0":
        return
    net = XORNet("tflite")
    net.eval()
    mge_result = dump_mge_model(net, net.data, tmp_file, True)
    _test_convert_result(net.data, tmp_file, mge_result, max_error, disable_nhwc=True)
