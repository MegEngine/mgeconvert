# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine
import numpy as np
import pytest
from mgeconvert.cambricon_converter.converter import CambriconConverter
from mgeconvert.mge_context import TopologyNetwork

from .quantization_utils import (
    QuantizationConvBnOpr,
    QuantizationLinearOpr,
    dump_mge_quantization_model,
)
from .utils import (
    ActiveOpr,
    BnOpr,
    ConcatOpr,
    ElemwiseOpr,
    PoolOpr,
    ReshapeOpr,
    SoftmaxOpr,
    SubtensorOpr,
    TransposeOpr,
    dump_mge_model,
)

max_error = 1e-4
tmp_file = "test_model"


def _test_convert_only(*_):
    toponet = TopologyNetwork("test_model.mge")
    batch_size = 1
    converter = CambriconConverter(toponet, batch_size, 1, "float32")
    converter.convert()
    converter.fuse()
    converter.dump(tmp_file + ".cambriconmodel")


def _test_convert_result(inputs, mge_results, max_err):
    toponet = TopologyNetwork("test_model.mge")
    batch_size = inputs.shape[0]
    converter = CambriconConverter(toponet, batch_size, 1, "float32")
    converter.convert()
    converter.fuse()
    cn_results = converter.forward(inputs).reshape(mge_results.shape)
    converter.dump(tmp_file + ".cambriconmodel")
    assert np.allclose(cn_results, mge_results, atol=max_err)


_test = _test_convert_only


@pytest.mark.parametrize("mode", ["relu", "tanh", "sigmoid"])
def test_active(mode):
    local_max_error = 1e-2  # tanh and sigmoid cannot pass 1e-4
    net = ActiveOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test(net.data, mge_result, local_max_error)


@pytest.mark.parametrize("mode", ["abs", "exp", "log"])
def test_elemwise_1(mode):
    local_max_error = 1e-3  # exp cannot pass 1e-4
    net = ElemwiseOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test(net.data, mge_result, local_max_error)


@pytest.mark.parametrize("mode", ["add", "sub", "mul", "cycle_div"])
def test_elemwise_2(mode):
    net = ElemwiseOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test(net.data, mge_result, max_error)


def test_softmax():
    local_max_error = 1e-2
    net = SoftmaxOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test(net.data, mge_result, local_max_error)


def test_transopse():
    # do not supoort transpose dimension "N".
    net = TransposeOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test(net.data, mge_result, max_error)


def test_concat():
    net = ConcatOpr()
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test(net.data, mge_result, max_error)


@pytest.mark.parametrize("mode", ["max", "avg"])
def test_pooling(mode):
    if megengine.__version__ > "0.6.0" and mode == "avg":
        return
    net = PoolOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test(net.data, mge_result, max_error)


def test_subtensor():
    net = SubtensorOpr(fix_batch=True)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test(net.data, mge_result, max_error)


def test_reshape():
    net = ReshapeOpr(fix_batch=True)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test(net.data, mge_result, max_error)


@pytest.mark.parametrize("mode", ["bn2d"])
def test_batchnorm(mode):
    net = BnOpr(mode)
    data = net.data1 if mode == "bn1d" else net.data2
    mge_result = dump_mge_model(net, data, tmp_file)
    _test(data, mge_result, max_error)


@pytest.mark.skip(reason="not trained")
def test_convbn2d():
    net = QuantizationConvBnOpr()
    mge_result = dump_mge_quantization_model(net, net.data, tmp_file)
    _test(net.data, mge_result, max_error)


@pytest.mark.skip(reason="not trained")
def test_linear():
    net = QuantizationLinearOpr()
    mge_result = dump_mge_quantization_model(net, net.data, tmp_file)
    _test(net.data, mge_result, max_error)


@pytest.mark.skip(reason="not trained")
def test_resnet18():
    """
    visit https://github.com/MegEngine/MegEngine for details.
    """
    return
