# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import List

import numpy as np

from ..mge_context import (
    ConcatOpr,
    ConvBiasForwardOpr,
    ConvolutionBackwardDataOpr,
    ConvolutionForwardOpr,
    MatrixMulOpr,
    TopologyNetwork,
    TransformerRule,
    optimize_for_conversion,
)
from ..mge_context.mge_utils import get_dtype_name, get_logger
from .cambricon_op import MGE2CN
from .lib import cnq
from .lib import operators as cnop
from .lib.model import Model
from .lib.tensor import DATA_TYPE, TENSOR_TYPE, Tensor

logger = get_logger(__name__)


map_dtype = {
    "float16": DATA_TYPE.FLOAT16,
    "float32": DATA_TYPE.FLOAT32,
}


class CambriconConverter:
    transformer_options: List[TransformerRule] = []

    def __init__(
        self,
        net,
        transformer_options=None,
        batch_size=4,
        core_number=1,
        data_type="float32",
        use_nhwc=False,
    ):
        if use_nhwc:
            Tensor.NCHW2NHWC = True
        Tensor._default_dtype = map_dtype[data_type]
        self.batch_size = batch_size
        self.core_number = core_number
        self.data_type = map_dtype[data_type]
        self.mge_net = net
        if transformer_options is not None:
            self.transformer_options = transformer_options
        optimize_for_conversion(self.mge_net, self.transformer_options)
        self.var_map = {}
        self.cn_inputs = []
        self.cn_outputs = []
        self.cn_oprs = []
        self.fusion = None

        self.check_model()

    def check_model(self):
        logger.info("check model...")
        unsupported_oprs = set()
        quantization_error_oprs = set()
        for opr in self.mge_net.all_oprs:
            if not isinstance(opr, tuple(MGE2CN.keys())):
                unsupported_oprs.add(type(opr))
            if isinstance(
                opr,
                (
                    ConvBiasForwardOpr,
                    ConvolutionBackwardDataOpr,
                    ConvolutionForwardOpr,
                    MatrixMulOpr,
                ),
            ):
                if (
                    get_dtype_name(opr.inp_vars[0]) != "QuantizedS8"
                    or get_dtype_name(opr.inp_vars[1]) != "QuantizedS8"
                    or (
                        len(opr.inp_vars) > 2
                        and get_dtype_name(opr.inp_vars[2]) != "QuantizedS32"
                    )
                    or get_dtype_name(opr.out_vars[0]) != "QuantizedS32"
                ):
                    quantization_error_oprs.add(type(opr))

        if unsupported_oprs:
            logger.error("Operators %s are not supported yet.", unsupported_oprs)
        if quantization_error_oprs:
            logger.error(
                "Operators %s should be quantized, "
                "check the function test_linear in test/test_cambricon for inspiration",
                quantization_error_oprs,
            )
        assert not unsupported_oprs and not quantization_error_oprs

    def get_cn_tensor(self, var):
        if var not in self.var_map:
            raise KeyError("can not find var {}".format(var.__dict__))
        return self.var_map[var]

    def set_cn_tensor(self, opr, var):
        ttype = TENSOR_TYPE.CONST
        dtype = self.data_type
        shp = var.shape
        data = var.np_data

        if var.qbit == "QuantizedS8":
            ttype = TENSOR_TYPE.FILTER
            dtype = DATA_TYPE.INT8
        if var.qbit == "QuantizedS32":
            data = data.astype(np.float32) * var.scale

        if len(shp) == 1:  # conv bias
            shp = (1, shp[0], 1, 1)
            data = data.reshape(shp)
        if len(shp) == 5:  # group conv's filter
            shp = (shp[0] * shp[1], shp[2], shp[3], shp[4])
            data = data.reshape(shp)
        # if var.qbit is None and shp[0] != 1:
        if isinstance(opr, ConcatOpr):
            shp = (self.batch_size,) + shp[1:]
            data = np.broadcast_to(data[:1], shp)
        return Tensor(
            shape=shp,
            ttype=ttype,
            dtype=dtype,
            data=data,
            scale=var.scale,
            name=opr.name,
        )

    def get_cn_inputs_and_outputs(self, opr):
        cn_inps = []
        for var in opr.inp_vars:
            if var not in self.var_map:
                self.var_map[var] = self.set_cn_tensor(opr, var)
            cn_inps.append(self.var_map[var])
        cn_oups = []
        for var in opr.out_vars:
            shp = (self.batch_size,) + var.shape[1:]
            self.var_map[var] = Tensor(shape=shp, name=var.name, scale=var.scale)
            cn_oups.append(self.var_map[var])
        return cn_inps, cn_oups

    def add_oprs(self, *cnoprs):
        self.cn_oprs.extend(cnoprs)

    def convert(self, end_op=None):
        for opr in self.mge_net.all_oprs:
            # Prune operators which calculate parameters.
            pruning = True
            for var in opr.out_vars:
                pruning = False if var.np_data is None else pruning
            if pruning:
                continue
            MGE2CN[type(opr)](opr, self)
            if opr.name == end_op:
                end_op = opr
                break
        assert not isinstance(end_op, str), (
            'This operator does not exist: "%s"' % end_op
        )
        self.cn_inputs = self.cn_oprs[0].inputs[0]
        if end_op is None:
            self.cn_outputs = self.cn_oprs[-1].outputs[0]
        else:
            self.cn_outputs = self.var_map[end_op.out_vars[0]]

    def fuse(self):
        self.fusion = cnop.Fusion("fusion", self.cn_inputs, self.cn_outputs)
        for cnopr in self.cn_oprs:
            self.fusion.fuse_op(cnopr)
        self.fusion.set_fusion_io()
        self.fusion.set_core_num(self.core_number)
        self.fusion.compile()

    def forward(self, feed_input: "np.ndarray"):
        self.cn_inputs.cpudata = feed_input
        self.cn_inputs.h2d()
        self.fusion.forward(cnq)
        return self.cn_outputs.cpudata

    def dump(self, fname):
        model = Model(self.fusion.name)
        model.add_fusionop(self.fusion)
        model.dump(fname)


def convert_to_cambricon(
    mge_fpath, filename, batch_size, core_number, data_type, use_nhwc
):
    """
    Convert megengine model to cambricon model.

    :param mge_fpath: the file path of megengine model.
    :type mge_fpath: str
    :param filename: cambricon model file name.
    :type filename: str
    :param batch_size: batch_size of the output cambricon model.
    :type batch_size: int
    :param core_number: core_number of the output cambricon model.
    :type core_number: int
    :param data_type: data_type of the output cambricon model, which should be
        "float32" or "float16".
    :type data_type: str
    :param use_nhwc: whether to use nhwc layout.
    :type use_nhwc: bool
    """
    assert isinstance(mge_fpath, str), "mge_fpath must be string"
    net = TopologyNetwork(mge_fpath)
    logger.info("init converter...")
    converter = CambriconConverter(
        net,
        batch_size=batch_size,
        core_number=core_number,
        data_type=data_type,
        use_nhwc=use_nhwc,
    )
    logger.info("convert operators to cambricon...")
    converter.convert()
    logger.info("%d operators converted...", len(converter.cn_oprs))
    converter.fuse()
    logger.info("fusing...")
    converter.dump(filename)
    logger.info("ok, dump model to %s", filename)
