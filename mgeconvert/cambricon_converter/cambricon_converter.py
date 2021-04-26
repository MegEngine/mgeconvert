# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from ..mge_context import (
    AxisAddRemoveOpr,
    BatchedMatrixMulOpr,
    BatchNormForwardOpr,
    BroadcastOpr,
    ConcatOpr,
    ConvBiasForwardOpr,
    ConvolutionBackwardDataOpr,
    ConvolutionForwardOpr,
    DimshuffleOpr,
    ElemwiseMultiTypeOpr,
    ElemwiseOpr,
    GetVarShapeOpr,
    Host2DeviceCopyOpr,
    IdentityOpr,
    MarkNoBroadcastElemwiseOpr,
    MatrixMulOpr,
    MultipleDeviceTensorHolderOpr,
    PoolingForwardOpr,
    ReduceOpr,
    ReshapeOpr,
    SharedDeviceTensorOpr,
    SubtensorOpr,
    TopologyNetwork,
    TypeCvtOpr,
    get_dtype_name,
    get_logger,
)
from .lib import cnq
from .lib import operators as cnop
from .lib.model import Model
from .lib.tensor import DATA_TYPE, TENSOR_TYPE, Tensor

logger = get_logger(__name__)
MGE2CN = {}


def _register_op(*oprs):
    def callback(impl):
        for opr in oprs:
            MGE2CN[opr] = impl
        return impl

    return callback


@_register_op(MultipleDeviceTensorHolderOpr)
def _multiple_device_tensor_holder(*_):
    pass


@_register_op(SharedDeviceTensorOpr)
def _shared_device_tensor(*_):
    pass


@_register_op(GetVarShapeOpr)
def _get_var_shape(*_):
    pass


@_register_op(Host2DeviceCopyOpr)
def _host2device_copy(opr, context):
    context.var_map[opr.out_vars[0]] = Tensor((context.batch_size,) + opr.shape[1:])


@_register_op(TypeCvtOpr)
def _typecvt(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    context.var_map[opr.out_vars[0]] = inps[0]
    if oups[0].scale is not None:
        inps[0].scale = oups[0].scale


@_register_op(ReshapeOpr)
def _reshape(opr, context):
    opr.inp_vars = opr.inp_vars[:1]
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    cnopr = cnop.Reshape(opr.name, inps, oups)
    context.add_oprs(cnopr)


@_register_op(BroadcastOpr)
def _broad_cast(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    cnopr = cnop.Broadcast(opr.name, inps, oups)
    context.add_oprs(cnopr)


@_register_op(DimshuffleOpr)
def _dimshuffle(opr, context):
    if opr.ndim == 1:
        inps, _ = context.get_cn_inputs_and_outputs(opr)
        context.var_map[opr.out_vars[0]] = inps[0]
    else:
        inps, oups = context.get_cn_inputs_and_outputs(opr)
        cnopr = cnop.Dimshuffle(opr.name, inps, oups, opr.pattern)
        context.add_oprs(cnopr)


@_register_op(ReduceOpr)
def _reduce(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    if opr.mode == "SUM_SQR":
        # Reduce(SUM_SQR) -> Elemwise(MUL) + Reduce(SUM)
        cnt_square = Tensor(shape=inps[0].shape)
        cn_square = cnop.Square(opr.name + "_SQUARE", inps, cnt_square)
        cn_sum = cnop.Reduce(opr.name + "_SUM", cnt_square, oups, opr.axis, "SUM")
        context.add_oprs(cn_square, cn_sum)
    else:
        cnopr = cnop.Reduce(opr.name, inps, oups, opr.axis, opr.mode)
        context.add_oprs(cnopr)


@_register_op(ConcatOpr)
def _concat(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    assert inps[1].shape[0] == inps[1].shape[0]
    cnopr = cnop.Concat(opr.name, inps, oups, opr.axis)
    context.add_oprs(cnopr)


@_register_op(AxisAddRemoveOpr)
def _axis_add_remove(opr, context):
    inps, _ = context.get_cn_inputs_and_outputs(opr)
    context.var_map[opr.out_vars[0]] = inps[0]


@_register_op(MarkNoBroadcastElemwiseOpr)
def _mark_no_broadcast_elemwise(opr, context):
    inps, _ = context.get_cn_inputs_and_outputs(opr)
    context.var_map[opr.out_vars[0]] = inps[0]


@_register_op(SubtensorOpr)
def _subtensor(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    inp_shape = inps[0].shape
    slc = [[0, inp_shape[i], 1] for i in range(4)]
    for i in range(len(opr.axis)):
        slc[opr.axis[i]] = [opr.begin_param[i], opr.end_param[i], opr.step_param[i]]
    cnopr = cnop.Slice(opr.name, inps, oups, slc)
    context.add_oprs(cnopr)


@_register_op(IdentityOpr)
def _identity(opr, context):
    inps, _ = context.get_cn_inputs_and_outputs(opr)
    context.var_map[opr.out_vars[0]] = inps[0]


@_register_op(ElemwiseOpr)
def _elemwise(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    if opr.mode == "FUSE_ADD_RELU":
        # FUSE_ADD_RELU -> ADD + RELU
        cnt_add = Tensor(shape=oups[0].shape)
        cn_add = cnop.Elemwise(opr.name + "_ADD", inps, cnt_add, "ADD")
        cn_relu = cnop.Active(opr.name + "_RELU", cnt_add, oups, "RELU")
        context.add_oprs(cn_add, cn_relu)
        return
    if opr.mode == "TRUE_DIV":
        # CNML does not support broadcast div.
        if inps[1].cpudata is not None:
            # TRUE_DIV -> BAISC_DIV + MUL
            cnt_basic_div = Tensor(
                shape=inps[1].shape, ttype=TENSOR_TYPE.CONST, data=1 / inps[1].cpudata
            )
            cn_mul = cnop.Elemwise(
                opr.name + "_MUL", [inps[0], cnt_basic_div], oups, "CYCLE_MUL"
            )
            context.add_oprs(cn_mul)
            return
        else:
            # real_div: support (n, c, h, w) / (1, 1, 1, 1) or (n, c, h, w) / (n, c, h, w)
            cnopr = cnop.Elemwise(opr.name, inps, oups, "TRUE_DIV")
            context.add_oprs(cnopr)
            return
    if opr.mode in ("NONE", "SIGMOID", "RELU", "TANH"):
        cnopr = cnop.Active(opr.name, inps, oups, opr.mode)
    elif opr.mode == "POW":
        if float(inps[1].cpudata) == 2:
            cnopr = cnop.Square(opr.name, inps[0], oups)
        elif float(inps[1].cpudata) == 0.5:
            cnopr = cnop.Sqrt(opr.name, inps[0], oups)
        else:
            raise NotImplementedError("power op in cambricon cannot work correctly")
    elif opr.mode in ("ADD", "MUL"):
        if inps[0].cpudata is None and inps[1].cpudata is None:
            cnopr = cnop.Elemwise(opr.name, inps, oups, opr.mode)
        else:
            if inps[0].cpudata is None:
                cnopr = cnop.Elemwise(opr.name, inps, oups, "CYCLE_" + opr.mode)
            else:
                cnopr = cnop.Elemwise(
                    opr.name, [inps[1], inps[0]], oups, "CYCLE_" + opr.mode
                )
    elif opr.mode == "NEGATE":
        inps.insert(0, Tensor((1, 1, 1, 1), TENSOR_TYPE.CONST, data=0))
        cnopr = cnop.Elemwise(opr.name, inps, oups, opr.mode)
    else:
        cnopr = cnop.Elemwise(opr.name, inps, oups, opr.mode)
    context.add_oprs(cnopr)


@_register_op(ElemwiseMultiTypeOpr)
def _elemwise_multitype(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    # QADD -> ADD + RELU
    cnt_add = Tensor(shape=oups[0].shape)
    cn_add = cnop.Elemwise(opr.name + "Add", inps, cnt_add, "ADD")
    cn_relu = cnop.Active(opr.name + "Relu", cnt_add, oups, "RELU")
    context.add_oprs(cn_add, cn_relu)


@_register_op(MatrixMulOpr)
def _matrix_mul(opr, context):
    # cnml does not support transposeB.
    if not opr.transposeB:
        var = opr.inp_vars[1]
        var.shape = (var.shape[1], var.shape[0])
        var.np_data = np.transpose(var.np_data)
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    cnopr = cnop.MatMul(opr.name, inps[0], oups)
    cnopr.param_dict["weight"] = inps[1]
    context.add_oprs(cnopr)


@_register_op(BatchedMatrixMulOpr)
def _batched_matrix_mul(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    cnopr = cnop.BatchMatMul(opr.name, inps, oups)
    context.add_oprs(cnopr)


@_register_op(BatchNormForwardOpr)
def _batch_norm(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    cnopr = cnop.BatchNorm(opr.name, inps[0], oups[2])
    cnopr.param_dict["running_mean"] = Tensor(
        shape=opr.mean.shape, ttype=TENSOR_TYPE.CONST, data=opr.mean
    )
    cnopr.param_dict["running_var"] = Tensor(
        shape=opr.var.shape, ttype=TENSOR_TYPE.CONST, data=opr.var
    )
    context.add_oprs(cnopr)


@_register_op(ConvolutionForwardOpr)
def _convolution(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    cnopr = cnop.Conv(
        opr.name,
        inps[0],
        oups,
        opr.sh,
        opr.sw,
        opr.dilation_h,
        opr.dilation_w,
        opr.ph,
        opr.pw,
        groups=opr.group,
    )
    cnopr.param_dict["W"] = inps[1]
    b_shp = (1, inps[1].shape[0], 1, 1)
    cnopr.param_dict["B"] = Tensor(
        shape=b_shp, ttype=TENSOR_TYPE.CONST, data=np.zeros(b_shp)
    )
    context.add_oprs(cnopr)


@_register_op(ConvBiasForwardOpr)
def _conv_bias(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    # ConvBias -> Convolution + Active
    cnt_convolution = Tensor(shape=oups[0].shape)
    # Convolution
    cn_convolution = cnop.Conv(
        opr.name + "Conv",
        inps[0],
        cnt_convolution,
        opr.sh,
        opr.sw,
        opr.dilation_h,
        opr.dilation_w,
        opr.ph,
        opr.pw,
        groups=opr.group,
    )
    cn_convolution.param_dict["W"] = inps[1]
    cn_convolution.param_dict["B"] = inps[2]
    # Active
    cn_active = cnop.Active(opr.name + "Act", cnt_convolution, oups, opr.activation)
    context.add_oprs(cn_convolution, cn_active)


@_register_op(ConvolutionBackwardDataOpr)
def _deconv(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    cn_deconv = cnop.Deconv(
        opr.name,
        inps[1],
        oups,
        opr.sh,
        opr.sw,
        opr.dilation_h,
        opr.dilation_w,
        opr.ph,
        opr.pw,
    )
    cn_deconv.param_dict["W"] = inps[0]
    b_shp = (1, inps[0].shape[0], 1, 1)
    cn_deconv.param_dict["B"] = Tensor(
        shape=b_shp, ttype=TENSOR_TYPE.CONST, data=np.zeros(b_shp)
    )
    context.add_oprs(cn_deconv)


@_register_op(PoolingForwardOpr)
def _pooling(opr, context):
    inps, oups = context.get_cn_inputs_and_outputs(opr)
    cnopr = cnop.Pool(
        opr.name,
        inps,
        oups,
        opr.kh,
        opr.kw,
        opr.sh,
        opr.sw,
        opr.ph,
        opr.pw,
        1,
        1,
        "MAX" if opr.mode == "MAX" else "AVG",
    )
    context.add_oprs(cnopr)


map_dtype = {
    "float16": DATA_TYPE.FLOAT16,
    "float32": DATA_TYPE.FLOAT32,
}


class CambriconConverter:
    def __init__(self, net, batch_size, core_number, data_type, use_nhwc=False):
        if use_nhwc:
            Tensor.NCHW2NHWC = True
        Tensor._default_dtype = map_dtype[data_type]
        self.batch_size = batch_size
        self.core_number = core_number
        self.data_type = map_dtype[data_type]
        self.mge_net = net
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
    converter = CambriconConverter(net, batch_size, core_number, data_type, use_nhwc)
    logger.info("convert operators to cambricon...")
    converter.convert()
    logger.info("%d operators converted...", len(converter.cn_oprs))
    converter.fuse()
    logger.info("fusing...")
    converter.dump(filename)
    logger.info("ok, dump model to %s", filename)
