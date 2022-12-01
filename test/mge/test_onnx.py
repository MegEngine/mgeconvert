# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
from test.utils import (
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
    TypeCvtOpr,
    XORNet,
    dump_mge_model,
)

import megengine as mge
import megengine.functional as F
import numpy as np
import onnxruntime as ort  # pylint: disable=import-error
import pytest
from megengine.jit import trace
from mgeconvert.backend.ir_to_onnx import OnnxConverter
from mgeconvert.converters.mge_to_onnx import mge_to_onnx
from mgeconvert.frontend.mge_to_ir import MGE_FrontEnd

max_error = 1e-6
tmp_file = "test_model"


def _test_convert_result(
    inputs, fpath, mge_result, max_err, min_version=7, max_version=12
):
    for version in range(min_version, max_version + 1):
        mge_to_onnx(
            fpath + ".mge", tmp_file + ".onnx", graph_name="graph", opset=version
        )

        onnx_net = ort.InferenceSession(tmp_file + ".onnx")
        input_name = onnx_net.get_inputs()[0].name
        X_test = inputs
        pred_onx = onnx_net.run(None, {input_name: X_test})[0]
        assert pred_onx.shape == mge_result.shape
        assert pred_onx.dtype == mge_result.dtype
        assert np.allclose(pred_onx, mge_result, atol=max_err)


@pytest.mark.parametrize("mode", ["normal", "group", "transpose"])
def test_conv2d(mode):
    net = ConvOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_conv_functional():
    if mge.__version__ < "1.2.0":
        return

    def convf(x, kernel):
        batch = int(kernel.shape[0])
        channel = int(kernel.shape[1])
        bc = batch * channel
        x = x.reshape((1, bc, int(x.shape[2]), int(x.shape[3])))
        kernel = kernel.reshape(bc, 1, 1, int(kernel.shape[2]), int(kernel.shape[3]))
        out = F.conv2d(x, kernel, groups=bc)
        out = out.reshape(batch, channel, int(out.shape[2]), int(out.shape[3]))
        return out

    @trace(symbolic=True, capture_as_const=True)
    def inference(x, kernel):
        output = convf(x, kernel)
        return output

    inpx = np.random.random((1, 48, 100, 100)).astype("float32")
    inpk = np.random.random((1, 48, 64, 64)).astype("float32")
    expect = inference(mge.tensor(inpx), mge.tensor(inpk)).numpy()
    inference.dump(
        tmp_file + ".mge", arg_names=["x", "kernel"], optimize_for_inference=False,
    )
    net = MGE_FrontEnd(tmp_file + ".mge").resolve()
    for version in range(8, 13):
        converter = OnnxConverter(net, opset_version=version, graph_name="graph")
        model = converter.convert()
        with open(tmp_file + ".onnx", "wb") as fout:
            fout.write(model.SerializeToString())
        onnx_net = ort.InferenceSession(tmp_file + ".onnx")
        pred_onx = onnx_net.run(None, {"x": inpx, "kernel": inpk})[0]
        assert pred_onx.shape == expect.shape
        assert pred_onx.dtype == expect.dtype
        assert np.allclose(pred_onx, expect, atol=1e-6)


def test_linear():
    net = LinearOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["bn1d", "bn2d"])
def test_batchnorm(mode):
    net = BnOpr(mode)
    net.eval()
    data = net.data1 if mode == "bn1d" else net.data2
    mge_result = dump_mge_model(net, data, tmp_file)
    _test_convert_result(data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["max", "avg"])
def test_pool(mode):
    if mode == "avg":
        return
    net = PoolOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


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


@pytest.mark.parametrize(
    "mode",
    ["add", "sub", "mul", "div", "abs", "exp", "log", "pow", "ceil", "floor", "max",],
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


@pytest.mark.parametrize("mode", ["sum", "max"])
def test_reduce(mode):
    net = ReduceOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


@pytest.mark.parametrize("mode", ["relu", "tanh", "sigmoid"])
def test_active(mode):
    net = ActiveOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


def test_broadcast():
    net = BroadcastOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error, min_version=8)


def test_typecvt():
    net = TypeCvtOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error, min_version=10)


@pytest.mark.skipif(
    mge.__version__ < "1.5.0",
    reason="MGE file for testing was dumped at version 1.5.0",
)
def test_convolutionbackwardfilter():
    import megengine.utils.comp_graph_tools as cgtools  # pylint: disable=import-outside-toplevel

    def infer_mge(x, file):
        infer_cg = cgtools.GraphInference(file + ".mge")  # pylint: disable=no-member
        y = list(infer_cg.run(x).values())[0]
        print(y.mean())
        return y

    file = os.path.join(os.path.dirname(__file__), "convolution-backward-filter")
    data = np.ones((8, 1, 32, 32), dtype=np.float32)
    mge_result = infer_mge(data, file)
    _test_convert_result(
        data, file, mge_result, max_error, min_version=8, max_version=8
    )


def test_xornet():
    if mge.__version__ < "1.1.0":
        return
    net = XORNet()
    mge_result = dump_mge_model(net, net.data, tmp_file, True)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


# skip
# @pytest.mark.parametrize(
#     "model",
#     [
#         "shufflenet_v2_x0_5",
#         "shufflenet_v2_x1_0",
#         "resnet18",
#         "resnet50",
#         "resnet101",
#         "resnext50_32x4d",
#     ],
# )
# def test_model(model):
#     data = (
#         np.random.randint(0, 255, 3 * 224 * 224)
#         .reshape((1, 3, 224, 224))
#         .astype(np.float32)
#     )
#     if mge.__version__ < "1.1.0":
#         commit_id = "dc2f2cfb228a135747d083517b98aea56e7aab92"
#     else:
#         commit_id = None
#     net = megengine.hub.load(
#         "megengine/models", model, use_cache=False, commit=commit_id, pretrained=True
#     )
#     mge_result = dump_mge_model(net, data, tmp_file)
#     _test_convert_result(data, tmp_file, mge_result, 1e-3)
