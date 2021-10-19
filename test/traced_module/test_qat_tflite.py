# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module,no-member

from test.traced_module.test_tflite import _test_convert_result
from test.utils import ConvOpr, LinearOpr

import megengine as mge
import megengine.module as M
import numpy as np
from megengine.core.tensor import dtype
from megengine.core.tensor.dtype import _builtin_quant_dtypes
from megengine.module.quant_dequant import QuantStub
from megengine.quantization.quantize import quantize_qat
from megengine.quantization.utils import create_qparams
from megengine.traced_module.fake_quant import FakeQuantize

from .tm_utils import get_traced_module

max_error = 1e-6
tmp_file = "test_model"


def get_qat_net(inp_dtype, net, num_inp=1, shape=(1, 16, 32, 32)):
    qat_net = quantize_qat(net)
    inps = []
    for _ in range(num_inp):
        data1 = mge.tensor(np.random.random(shape)) * 16
        data1 = data1.astype(inp_dtype)
        inp1 = mge.tensor(dtype.convert_from_qint8(data1.numpy()))
        inp1.qparams.scale = mge.tensor(dtype.get_scale(inp_dtype))
        inp1.qparams.dtype_meta = dtype._builtin_quant_dtypes["qint8"]
        inps.append(inp1)
    return qat_net, inps


def test_qat_conv_qint8():
    class QConvOpr(M.Module):
        def __init__(self):
            super().__init__()
            self.normal_conv = M.Conv2d(
                3, 30, 3, stride=(2, 3), padding=(3, 1), dilation=(2, 2),
            )
            self.normal_conv.bias = mge.Parameter(
                np.random.random(self.normal_conv.bias.shape).astype(np.float32)
            )

        def forward(self, x):
            x = self.normal_conv(x)
            return x

    net = QConvOpr()
    qat_net = quantize_qat(net)

    inp_dtype = dtype.qint8(16.0 / 128)
    data = mge.tensor(np.random.random((1, 3, 224, 224))) * 16
    data = data.astype(inp_dtype)
    inp = mge.tensor(dtype.convert_from_qint8(data.numpy()))
    inp.qparams.scale = mge.tensor(dtype.get_scale(inp_dtype))
    inp.qparams.dtype_meta = dtype._builtin_quant_dtypes["qint8"]

    traced_module, tm_result = get_traced_module(qat_net, inp)
    print(traced_module.flatten().graph)
    inp = inp.astype(inp_dtype)
    out_dtype = traced_module.graph.outputs[0].qparams
    scale = out_dtype.scale.numpy()
    _test_convert_result(
        inp, traced_module, tm_result, scale=scale, require_quantize=True
    )


def test_deconv_qint8():
    net = ConvOpr("tflite_transpose")
    qat_net = quantize_qat(net)

    inp_dtype = dtype.qint8(16.0 / 128)
    data = mge.tensor(np.random.random((1, 3, 64, 64))) * 16
    data = data.astype(inp_dtype)
    inp = mge.tensor(dtype.convert_from_qint8(data.numpy()))
    inp.qparams.scale = mge.tensor(dtype.get_scale(inp_dtype))
    inp.qparams.dtype_meta = dtype._builtin_quant_dtypes["qint8"]

    traced_module, tm_result = get_traced_module(qat_net, inp)
    print(traced_module.flatten().graph)
    inp = inp.astype(inp_dtype)
    out_dtype = traced_module.graph.outputs[0].qparams
    scale = out_dtype.scale.numpy()
    _test_convert_result(
        inp, traced_module, tm_result, scale=scale, require_quantize=True
    )


def test_linear():
    net = LinearOpr()
    inp_dtype = dtype.qint8(16.0 / 128.0)
    qat_net, inps = get_qat_net(inp_dtype, net, shape=(10, 100))
    traced_module, tm_result = get_traced_module(qat_net, inps[0])
    print(traced_module.flatten().graph)
    out_dtype = traced_module.graph.outputs[0].qparams
    scale = out_dtype.scale.numpy()
    inp = inps[0].astype(inp_dtype)
    _test_convert_result(
        inp, traced_module, tm_result, scale=scale, require_quantize=True
    )


def test_add():
    class ElemwiseOpr(M.Module):
        def __init__(self,):
            super().__init__()
            self.data = np.ones((2, 3, 224, 224)).astype(np.float32)
            self.data1 = np.random.random((1, 3, 1, 1)).astype(np.float32)
            self.add1 = M.Elemwise("add")
            self.add2 = M.Elemwise("add")
            self.add3 = M.Elemwise("add")

            scale = mge.tensor((16.0 / 128.0))
            self.quant_stub = QuantStub()
            self.quant_stub.act_fake_quant = FakeQuantize(
                _builtin_quant_dtypes["qint8"]
            )
            self.quant_stub.act_fake_quant.set_qparams(
                create_qparams(
                    dtype_meta=_builtin_quant_dtypes["qint8"],
                    scale=scale,
                    zero_point=None,
                )
            )
            self.quant_stub1 = QuantStub()
            self.quant_stub1.act_fake_quant = FakeQuantize(
                _builtin_quant_dtypes["qint8"]
            )
            self.quant_stub1.act_fake_quant.set_qparams(
                create_qparams(
                    dtype_meta=_builtin_quant_dtypes["qint8"],
                    scale=scale,
                    zero_point=None,
                )
            )

        def forward(self, a):
            n = self.quant_stub(mge.tensor(np.float32(10)))
            data1 = self.quant_stub1(mge.tensor(self.data1))
            x = self.add1(a, n)
            y = self.add2(a, data1)
            z = self.add3(x, y)
            return z

    net = ElemwiseOpr()
    inp_dtype = dtype.qint8(16.0 / 128.0)
    qat_net, inps = get_qat_net(inp_dtype, net, shape=(1, 3, 1, 1))
    traced_module, tm_result = get_traced_module(qat_net, inps[0])
    print(traced_module.flatten().graph)
    out_dtype = traced_module.graph.outputs[0].qparams
    scale = out_dtype.scale.numpy()
    inp = inps[0].astype(inp_dtype)
    _test_convert_result(
        inp, traced_module, tm_result, scale=scale, require_quantize=True
    )
