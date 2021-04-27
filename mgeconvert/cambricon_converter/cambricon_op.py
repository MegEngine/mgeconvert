# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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
    TypeCvtOpr,
)
from .lib import operators as cnop
from .lib.tensor import TENSOR_TYPE, Tensor

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
