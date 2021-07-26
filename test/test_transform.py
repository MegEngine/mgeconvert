# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine as mge
from megengine.jit import trace
from mgeconvert.mge_context import (
    TopologyNetwork,
    TransformerRule,
    optimize_for_conversion,
)

from .utils import ActiveOpr, ConvOpr, LinearOpr


def dump_mge_model(net, data, fpath="test_model.mge", optimize_for_inference=False):
    if mge.__version__ <= "0.6.0":

        @trace(symbolic=True)
        def inference(data, *, net):
            net.eval()
            output = net(data)
            return output

        inference.trace(data, net=net)
        mge_result = inference(data, net=net).numpy()
        inference.dump(
            fpath, arg_names=["data"], optimize_for_inference=optimize_for_inference,
        )
        return mge_result
    else:
        mge_result = net(mge.tensor(data))
        net.eval()
        mge_result = net(mge.tensor(data))

        @trace(symbolic=True, capture_as_const=True)
        def inference(data):
            net.eval()
            output = net(data)
            return output

        inference(mge.tensor(data))
        inference.dump(
            fpath, arg_names=["data"], optimize_for_inference=optimize_for_inference,
        )
        return mge_result.numpy()


def test_fuse_for_leaky_relu():
    net = ActiveOpr(mode="leaky_relu")
    dump_mge_model(net, net.data, fpath="test_model.mge")
    net = TopologyNetwork("test_model.mge")
    optimize_for_conversion(net, TransformerRule.FUSE_FOR_LEAKY_RELU)
    actual = list(type(opr).__name__ for opr in net.all_oprs)
    desired = ["Host2DeviceCopyOpr", "LeakyReluOpr"]
    assert actual == desired


def test_fuse_for_conv_bias():
    net = ConvOpr(mode="normal")
    dump_mge_model(net, net.data, fpath="test_model.mge")
    net = TopologyNetwork("test_model.mge")
    optimize_for_conversion(net, TransformerRule.FUSE_FOR_CONV_BIAS)
    actual = list(type(opr).__name__ for opr in net.all_oprs)
    desired = ["Host2DeviceCopyOpr", "ConvForwardBiasOpr"]
    assert actual == desired


def test_fuse_for_deconv_bias():
    net = ConvOpr(mode="transpose")
    dump_mge_model(net, net.data, "test_model.mge")
    net = TopologyNetwork("test_model.mge")
    optimize_for_conversion(net, TransformerRule.FUSE_FOR_DECONV_BIAS)
    actual = list(type(opr).__name__ for opr in net.all_oprs)
    desired = [
        "Host2DeviceCopyOpr",
        "ConvolutionBackwardDataBiasOpr",
        "ConvolutionBackwardDataBiasOpr",
    ]
    assert actual == desired


def test_fuse_for_fully_connected():
    net = LinearOpr()
    dump_mge_model(net, net.data, "test_model.mge")
    net = TopologyNetwork("test_model.mge")
    optimize_for_conversion(net, TransformerRule.FUSE_ACTIVATION)
    optimize_for_conversion(net, TransformerRule.FUSE_FOR_FULLY_CONNECTED)
    assert net.all_oprs[-1].activation == "RELU"
    actual = list(type(opr).__name__ for opr in net.all_oprs)
    desired = ["Host2DeviceCopyOpr", "MatrixMulOpr", "FullyConnectedOpr"]
    assert actual == desired
