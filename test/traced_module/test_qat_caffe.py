# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module,no-member

from test.utils import LinearOpr

import megengine as mge
import megengine.module as M
import numpy as np
from megengine.core.tensor import dtype
from megengine.core.tensor.dtype import _builtin_quant_dtypes
from megengine.module.quant_dequant import QuantStub
from megengine.quantization.quantize import quantize_qat
from megengine.quantization.utils import create_qparams
from megengine.traced_module.fake_quant import FakeQuantize

from .test_caffe import _test_convert_result
from .tm_utils import get_traced_module

max_err = 1e-6


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


def get_qat_inputs_quint8(inp_dtype, num_inp=1, shape=(1, 16, 384, 512)):
    inps = []
    for _ in range(num_inp):
        data1 = mge.tensor(np.random.random(shape)) * 16
        data1 = data1.astype(inp_dtype)
        inp1 = mge.tensor(dtype.convert_from_quint8(data1.numpy()))
        inp1.qparams.scale = mge.tensor(dtype.get_scale(inp_dtype))
        inp1.qparams.zero_point = mge.tensor(dtype.get_zero_point(inp_dtype))
        inp1.qparams.dtype_meta = dtype._builtin_quant_dtypes["quint8"]
        inps.append(inp1)
    return inps


def test_linear():
    net = LinearOpr()
    inp_dtype = dtype.qint8(16.0 / 128.0)
    qat_net, inps = get_qat_net(inp_dtype, net, shape=(10, 100))
    traced_module, tm_result = get_traced_module(qat_net, inps[0])
    inp = inps[0].astype(inp_dtype)
    _test_convert_result(inp, traced_module, tm_result, max_err, require_quantize=True)


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
    inp = inps[0].astype(inp_dtype)
    _test_convert_result(
        inp,
        traced_module,
        tm_result,
        max_err,
        require_quantize=True,
        split_conv_relu=True,
    )


def test_det_model():
    net = mge.load("models_fire_det.fix_batch.fuse_scale_cpu.pkl")
    inp_dtype = dtype.qint8(16.0 / 128.0)
    qat_net, inps = get_qat_net(inp_dtype, net, shape=(1, 3, 512, 512))
    traced_module, tm_result = get_traced_module(qat_net, inps[0])
    inp = inps[0].astype(inp_dtype)
    _test_convert_result(inp, traced_module, tm_result, max_err, require_quantize=True)


def test_snpe_model_8f():
    model = "8w16f_backbone.tm"
    net = mge.load(model)
    print(net.flatten().graph)
    inp_dtype = dtype.quint8(16.0 / 128.0, 128)
    inps = get_qat_inputs_quint8(inp_dtype, num_inp=2, shape=(1, 16, 384, 512))
    tm_result = dict(zip(net.graph.outputs, net(*inps)))
    _test_convert_result(
        inps,
        net,
        tm_result,
        max_err,
        input_data_type="quint8",
        input_scales=inps[0].qparams.scale,
        input_zero_points=inps[0].qparams.zero_point,
        require_quantize=True,
        param_fake_quant=True,
        split_conv_relu=True,
        input_name=["inp", "prev"],
    )
