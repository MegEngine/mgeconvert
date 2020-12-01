# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from functools import partial

import megengine as mge
import megengine.module as M
import numpy as np
from megengine.jit import trace
from megengine.quantization.fake_quant import FakeQuantize
from megengine.quantization.observer import MinMaxObserver
from megengine.quantization.qconfig import QConfig
from megengine.quantization.quantize import quantize, quantize_qat


def dump_mge_quantization_model(net, data, fpath="test_model"):
    net = qat_cn(net)
    net = quantize(net)
    if mge.__version__ <= "0.6.0":

        @trace(symbolic=True)
        def inference(data, *, net):
            net.eval()
            output = net(data)
            return output

        inference.trace(data, net=net)
        mge_result = inference(data, net=net).numpy()
        inference.dump(fpath + ".mge", arg_names=["data"], optimize_for_inference=True)
        return mge_result
    else:

        @trace(symbolic=True, capture_as_const=True)
        def inference(data):
            net.eval()
            output = net(data)
            return output

        mge_result = inference(mge.tensor(data))
        inference.dump(fpath + ".mge", optimize_for_inference=False)
        return mge_result


class QuantizationLinearOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((10, 100)).astype(np.float32)
        self.quant = M.QuantStub()
        self.linear = M.Linear(100, 200, bias=False)
        self.dequant = M.DequantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.linear(x)
        x = self.dequant(x)
        return x


class QuantizationConvBnOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((1, 3, 224, 224)).astype(np.float32)
        self.quant = M.QuantStub()
        self.dequant = M.DequantStub()
        self.conv_bn = M.ConvBn2d(3, 30, 3, 1, 1)
        self.group_conv_bn = M.ConvBn2d(3, 30, 3, 1, 1, 1, 3)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv_bn(x)
        x = self.dequant(x)
        return x


def qat_cn(module, inplace=True):
    new_mod = quantize_qat(
        module, inplace=inplace, qconfig=set_parameter_quantization_config
    )
    for submodule in new_mod.modules():
        if isinstance(submodule, M.qat.QuantStub):
            submodule.set_qconfig(set_activation_quantization_config)
    return new_mod


set_activation_quantization_config = QConfig(
    weight_observer=None,
    act_observer=partial(MinMaxObserver, dtype="qint8"),
    weight_fake_quant=None,
    act_fake_quant=partial(FakeQuantize, dtype="qint8"),
)

set_parameter_quantization_config = QConfig(
    weight_observer=partial(MinMaxObserver, dtype="qint8"),
    act_observer=None,
    weight_fake_quant=partial(FakeQuantize, dtype="qint8"),
    act_fake_quant=None,
)
