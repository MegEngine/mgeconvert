# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
from math import ceil

import numpy as np

from ..mge_context import (
    AxisAddRemoveOpr,
    BatchNormForwardOpr,
    ConcatOpr,
    ConvolutionForwardOpr,
    DimshuffleOpr,
    ElemwiseOpr,
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
    Tensor,
    get_logger,
)
from ..mge_context.mge_utils import get_symvar_value, isconst
from .caffe_pb import caffe_pb2 as cp  # pylint: disable=import-error

logger = get_logger(__name__)
MGE2CAFFE = {}


def _register_op(*oprs):
    def callback(impl):
        for opr in oprs:
            MGE2CAFFE[opr] = impl
        return impl

    return callback


def silence_blob(blob):
    return cp.LayerParameter(
        name="{}_silence".format(blob), type="Silence", bottom=[blob]
    )


def _gen_layer(opr, etype, context, single_input=True, **kwargs):
    bottom = (
        [context.get_blob_name(opr.inp_vars[0])]
        if single_input
        else list(map(context.get_blob_name, opr.inp_vars))
    )
    top = [context.set_blob_name(opr.out_vars[0], opr.name)]
    return cp.LayerParameter(
        bottom=bottom, top=top, name=opr.name, type=etype, **kwargs
    )


def pooling_layer(name, bottom, top, mode, ph, pw, sh, sw, kh, kw):
    param = cp.PoolingParameter(
        pool=0 if mode == "MAX" else 1,
        pad_h=ph,
        pad_w=pw,
        stride_h=sh,
        stride_w=sw,
        kernel_h=kh,
        kernel_w=kw,
    )
    return cp.LayerParameter(
        bottom=bottom, top=top, name=name, type="Pooling", pooling_param=param
    )


def _broadcast_for_eltwiseopr(oprname, InputA, InputB, context):
    shapeA = InputA.shape
    topA = [context.get_blob_name(InputA)]
    ndimA = InputA.ndim
    if not isconst(InputB):
        shapeB = InputB.shape
        topB = [context.get_blob_name(InputB)]
        ndimB = InputB.ndim
    else:
        shapeB = InputB.shape
        topB = InputB.np_data.copy()
        ndimB = topB.ndim
    if ndimA != ndimB:
        if ndimA < ndimB:
            shapeA = (1,) * (ndimB - ndimA) + shapeA
            ndimA = ndimB
            param = cp.ReshapeParameter(shape=cp.BlobShape(dim=shapeA))
            context.add_layer(
                cp.LayerParameter(
                    bottom=topA,
                    top=[topA[0] + "_reshape"],
                    name=oprname + context.gen_name,
                    type="Reshape",
                    reshape_param=param,
                )
            )
            topA = [topA[0] + "_reshape"]
        else:
            shapeB = (1,) * (ndimA - ndimB) + shapeB
            ndimB = ndimA
            if not isconst(InputB):
                param = cp.ReshapeParameter(shape=cp.BlobShape(dim=shapeB))
                context.add_layer(
                    cp.LayerParameter(
                        bottom=topB,
                        top=[topB[0] + "_reshape"],
                        name=oprname + context.gen_name,
                        type="Reshape",
                        reshape_param=param,
                    )
                )
                topB = [topB[0] + "_reshape"]
            else:
                topB = topB.reshape(shapeB)

    assert ndimA == ndimB
    shape = []
    for i in range(ndimA):
        shpA, shpB = shapeA[i], shapeB[i]
        if shpA == shpB:
            shape.append(shpA)
            continue
        if shpA == 1:
            shape.append(shpB)
            bottom = topA
            name = oprname + context.gen_name
            topA = [name]
            logger.warning("Add 'tile layers' for broadcast")
            context.add_layer(
                cp.LayerParameter(
                    name=name,
                    type="Tile",
                    bottom=bottom,
                    top=topA,
                    tile_param=cp.TileParameter(axis=i, tiles=shpB),
                )
            )
        else:
            assert shpB == 1
            shape.append(shpA)
            if not isconst(InputB):
                bottom = topB
                name = oprname + context.gen_name
                topB = [name]
                logger.warning("Add 'tile layers' for broadcast")
                context.add_layer(
                    cp.LayerParameter(
                        name=name,
                        type="Tile",
                        bottom=bottom,
                        top=topB,
                        tile_param=cp.TileParameter(axis=i, tiles=shpA),
                    )
                )
            else:
                topB = np.repeat(topB, shpA, i)
    return (topA, topB, shape)


@_register_op(DimshuffleOpr)
def _dimshufulle(opr, context):
    def swap_two_dimension(x, y, f, top, context):
        prefix = "{}_swap_{}_{}".format(opr.name, x, y)
        if opr.inp_vars[0].shape[f[x]] > 1:
            bottom = top
            top = [context.gen_name for _ in range(opr.inp_vars[0].shape[f[x]])]
            context.add_layer(
                cp.LayerParameter(
                    name="{}_slice_x".format(prefix),
                    type="Slice",
                    bottom=bottom,
                    top=top,
                    slice_param=cp.SliceParameter(
                        axis=x, slice_point=list(range(1, opr.inp_vars[0].shape[f[x]]))
                    ),
                )
            )
        if opr.inp_vars[0].shape[f[y]] > 1:
            bottom = top
            top = [
                [context.gen_name for _ in range(opr.inp_vars[0].shape[f[y]])]
                for _ in range(opr.inp_vars[0].shape[f[x]])
            ]
            context.add_layer(
                list(
                    map(
                        lambda z: cp.LayerParameter(
                            name="{}_slice_x{}_slice_y".format(prefix, z),
                            type="Slice",
                            bottom=[bottom[z]],
                            top=top[z],
                            slice_param=cp.SliceParameter(
                                axis=y,
                                slice_point=list(range(1, opr.inp_vars[0].shape[f[y]])),
                            ),
                        ),
                        range(opr.inp_vars[0].shape[f[x]]),
                    )
                )
            )
            bottom = top
            top = [context.gen_name for _ in range(opr.inp_vars[0].shape[f[x]])]
            context.add_layer(
                list(
                    map(
                        lambda z: cp.LayerParameter(
                            name="{}_concat_x{}".format(prefix, z),
                            type="Concat",
                            bottom=bottom[z],
                            top=[top[z]],
                            concat_param=cp.ConcatParameter(axis=x),
                        ),
                        range(opr.inp_vars[0].shape[f[x]]),
                    )
                )
            )
        if opr.inp_vars[0].shape[f[x]] > 1:
            bottom = top
            top = [context.gen_name]
            context.add_layer(
                cp.LayerParameter(
                    name="{}_concat_y".format(prefix),
                    type="Concat",
                    bottom=bottom,
                    top=top,
                    concat_param=cp.ConcatParameter(axis=y),
                )
            )
        f[x], f[y] = f[y], f[x]
        return top

    logger.warning("Add 'slice/concat layers' in operator: Dimshuffle")

    l = list(filter(lambda x: x != "x", list(opr.pattern)))
    assert len(l) == opr.inp_vars[0].ndim
    nl = []
    for i, _ in enumerate(l):
        while l[i] != i:
            j = l[i]
            nl.append((i, j))
            l[i], l[j] = l[j], l[i]
    nl.reverse()

    f = list(range(len(l)))

    top = [context.get_blob_name(opr.inp_vars[0])]
    for i in nl:
        top = swap_two_dimension(i[0], i[1], f, top, context)
    if len(opr.pattern) != opr.inp_vars[0].ndim:
        bottom = top
        top = [context.gen_name]
        context.add_layer(
            cp.LayerParameter(
                name="{}_reshape".format(opr.name),
                type="Reshape",
                bottom=bottom,
                top=top,
                reshape_param=cp.ReshapeParameter(shape=cp.BlobShape(dim=opr.shapes)),
            )
        )
    context.set_blob_name(opr.out_vars[0], list(top)[0])


@_register_op(SubtensorOpr)
def _subtensor(opr: SubtensorOpr, context):
    logger.warning("Add 'slice/concat layers' in operator: Subtensor")

    top = [context.get_blob_name(opr.inp_vars[0])]

    def axis_suffix_gen():
        yield ""
        i = 2
        while True:
            yield "_axis_{}".format(i)
            i += 1

    axis_suffixes = axis_suffix_gen()

    def get_slice_list(n, s: slice, need_reverse=False):
        l = np.arange(n)[s]
        v = np.zeros(n + 1, dtype=bool)
        if need_reverse:
            v[l] |= True
            v[l + 1] |= True
        else:
            v[l] ^= True
            v[l + 1] ^= True
        v[0] = False
        return list(np.arange(n)[v[:-1]])

    def get_concat_list(n, axis, s: slice):
        l = np.arange(n)[s]
        assert len(l) > 0, "got empty tensor in opr: {}, axis: {}.".format(opr, axis)
        return list(l[np.insert(l[1:] - l[:-1] != 1, 0, True)])

    def top_name(k, concat_list, name):
        if k in concat_list and len(concat_list) == 1:
            return name
        return name + context.gen_name

    for i in range(0, len(opr.begin_param)):
        begin = opr.begin_param[i]
        end = opr.end_param[i]
        step = opr.step_param[i]
        axis = opr.axis[i]
        sl = slice(begin, end, step)

        n = opr.inp_vars[0].shape[axis]

        slice_list = get_slice_list(n, sl, step < 0)
        concat_list = get_concat_list(n, axis, sl)

        if slice_list == [] or concat_list == []:
            continue

        bottom = top
        name = opr.name + next(axis_suffixes)

        spldict = collections.OrderedDict(
            [(k, top_name(k, concat_list, name)) for k in [0] + slice_list]
        )
        top = list(spldict.values())
        context.add_layer(
            cp.LayerParameter(
                name=name,
                type="Slice",
                bottom=bottom,
                top=top,
                slice_param=cp.SliceParameter(axis=axis, slice_point=slice_list),
            )
        )
        if len(concat_list) > 1:
            bottom = list(map(lambda x, spl=spldict: spl[x], concat_list))  # type: ignore[misc]
            name = name + "_concat"
            top = [name]
            context.add_layer(
                cp.LayerParameter(
                    name=name,
                    type="Concat",
                    bottom=bottom,
                    top=top,
                    concat_param=cp.ConcatParameter(axis=axis),
                )
            )
        else:
            top = [spldict[concat_list[0]]]

        context.add_layer(
            list(
                map(
                    lambda _, spl=spldict: silence_blob(spl[_]),  # type: ignore[misc]
                    [_ for _ in [0] + slice_list if _ not in concat_list],
                )
            )
        )

    context.set_blob_name(opr.out_vars[0], list(top)[0])

    if len(opr.squeeze_axis):
        name = opr.name + "_reshape"
        param = cp.ReshapeParameter(shape=cp.BlobShape(dim=opr.out_vars[0].shape))
        bottom = top
        top = [context.reset_blob_name(opr.out_vars[0], name)]
        context.add_layer(
            cp.LayerParameter(
                name=name, type="Reshape", bottom=bottom, top=top, reshape_param=param
            )
        )
        logger.warning("Add 'reshape layers' in operator: Subtensor")


@_register_op(MultipleDeviceTensorHolderOpr, SharedDeviceTensorOpr)
def _(*_):
    pass


@_register_op(Host2DeviceCopyOpr)
def _data_provider(opr, context):
    param = cp.InputParameter(shape=[cp.BlobShape(dim=opr.shape)])
    context.add_layer(
        _gen_layer(opr, "Input", context, single_input=False, input_param=param)
    )


def _arith(opr, mode, context):
    atype = mode
    if isconst(opr.inp_vars[0]) and isconst(opr.inp_vars[1]):
        return
    elif isconst(opr.inp_vars[0]) or isconst(opr.inp_vars[1]):
        if isconst(opr.inp_vars[0]):
            inpA = opr.inp_vars[1]
            const = opr.inp_vars[0]
            order = 1
        else:
            inpA = opr.inp_vars[0]
            const = opr.inp_vars[1]
            order = 0
        topA, topB, shape = _broadcast_for_eltwiseopr(opr.name, inpA, const, context)

        layer_param = cp.ScaleParameter(axis=len(shape) - topB.ndim, num_axes=topB.ndim)
        if atype in {"ADD", "SUB"}:
            layer_param.bias_term = True
            param_b = topB
            param_k = np.ones(shape=param_b.shape)
            if atype == "SUB":
                if order == 0:
                    param_b = -param_b  # pylint: disable=invalid-unary-operand-type
                else:
                    param_k = -param_k
            blobs = [context.gen_blob_proto(param_k), context.gen_blob_proto(param_b)]
        else:
            param_k = topB
            if atype == "TRUE_DIV":
                if order == 0:
                    param_k = 1.0 / param_k
                else:
                    bottom = topA
                    name = opr.name + context.gen_name
                    topA = [name]
                    context.add_layer(
                        cp.LayerParameter(
                            name=name,
                            type="Power",
                            bottom=bottom,
                            top=topA,
                            power_param=cp.PowerParameter(scale=1, shift=0, power=-1),
                        )
                    )
            blobs = [context.gen_blob_proto(param_k)]
        bottom = topA
        top = [context.set_blob_name(opr.out_vars[0], opr.name)]
        context.add_layer(
            cp.LayerParameter(
                name=opr.name,
                type="Scale",
                bottom=bottom,
                top=top,
                scale_param=layer_param,
                blobs=blobs,
            )
        )
    else:
        topA, topB, _ = _broadcast_for_eltwiseopr(
            opr.name, opr.inp_vars[0], opr.inp_vars[1], context
        )
        if atype in {"ADD", "SUB"}:
            param = cp.EltwiseParameter(operation="SUM")
            if atype == "SUB":
                param.coeff.extend([1, -1])
        else:
            if atype == "TRUE_DIV":
                bottom = topB
                name = opr.name + context.gen_name
                topB = [name]
                context.add_layer(
                    cp.LayerParameter(
                        name=name,
                        type="Power",
                        bottom=bottom,
                        top=topB,
                        power_param=cp.PowerParameter(scale=1, shift=0, power=-1),
                    )
                )
            param = cp.EltwiseParameter(operation="PROD")
        bottom = topA + topB
        top = [context.set_blob_name(opr.out_vars[0], opr.name)]
        context.add_layer(
            cp.LayerParameter(
                name=opr.name,
                type="Eltwise",
                bottom=bottom,
                top=top,
                eltwise_param=param,
            )
        )


@_register_op(ElemwiseOpr, MarkNoBroadcastElemwiseOpr, IdentityOpr)
def _eltwise(opr: ElemwiseOpr, context):
    atype = opr.mode
    if atype == "Identity":
        context.set_blob_name(opr.out_vars[0], context.get_blob_name(opr.inp_vars[0]))
    elif atype == "FUSE_ADD_RELU":
        _arith(opr, "ADD", context)
        bottom = [context.get_blob_name(opr.out_vars[0])]
        top = [context.reset_blob_name(opr.out_vars[0], bottom[0] + ":relu")]
        context.add_layer(
            cp.LayerParameter(bottom=bottom, top=top, name=top[0], type="ReLU")
        )
    elif atype == "FUSE_MUL_ADD3":

        mge_opr = opr._opr
        mul_opr = ElemwiseOpr(mge_opr)
        mul_opr.name = mul_opr.name + "_MUL"
        mul_opr.add_inp_var(opr.inp_vars[0])
        mul_opr.add_inp_var(opr.inp_vars[1])
        mul_out = Tensor(opr.out_vars[0]._var, mge_opr)
        mul_opr.add_out_var(mul_out)
        _arith(mul_opr, "MUL", context)

        add_opr = ElemwiseOpr(mge_opr)
        add_opr.name = add_opr.name + "_ADD"
        add_opr.add_inp_var(mul_out)
        add_opr.add_inp_var(opr.inp_vars[2])
        add_opr.add_out_var(opr.out_vars[0])
        _arith(add_opr, "ADD", context)

    elif atype == "RELU":
        context.add_layer(_gen_layer(opr, "ReLU", context))
    elif atype == "TANH":
        context.add_layer(_gen_layer(opr, "TanH", context))
    elif atype == "EXP":
        context.add_layer(_gen_layer(opr, "Exp", context))
    elif atype == "SIGMOID":
        context.add_layer(_gen_layer(opr, "Sigmoid", context))
    elif atype == "LOG":
        context.add_layer(_gen_layer(opr, "Log", context))
    elif atype == "ABS":
        context.add_layer(_gen_layer(opr, "AbsVal", context))
    elif atype in ["ADD", "SUB", "MUL", "TRUE_DIV"]:
        _arith(opr, opr.mode, context)
    else:
        assert (
            False
        ), "Elemwise op doesn't support mode {}, you can implement it in _eltwise".format(
            opr.mode
        )


@_register_op(ConvolutionForwardOpr)
def _convolution(opr, context):
    ph, pw = opr.ph, opr.pw
    sh, sw = opr.sh, opr.sw
    kh, kw = opr.kh, opr.kw
    param_W = opr.param_W
    group = opr.group
    dilation_h = opr.dilation_h
    param_W = param_W.reshape((-1,) + param_W.shape[-3:])
    bias_term = opr.bias_term
    assert bias_term == False
    blobs = [
        context.gen_blob_proto(param_W),
    ]
    param = cp.ConvolutionParameter(
        num_output=opr.num_output,
        pad_h=ph,
        pad_w=pw,
        stride_h=sh,
        stride_w=sw,
        kernel_h=kh,
        kernel_w=kw,
        dilation=[dilation_h],
        group=group,
        bias_term=bias_term,
    )

    context.add_layer(
        _gen_layer(opr, "Convolution", context, convolution_param=param, blobs=blobs)
    )


def _biasadder(opr, context):
    assert opr.param_manager["b"].owner_opr.is_value_initialized
    param_b = opr.param_manager["b"].owner_opr.get_value()

    if opr.brdcst_mode == 0:
        param_b = param_b[0]
        param = cp.BiasParameter(axis=0, num_axes=0)
    elif opr.brdcst_mode == 1:
        param = cp.BiasParameter(axis=1, num_axes=1)
    else:
        param = cp.BiasParameter(axis=1, num_axes=-1)

    blobs = [
        context.gen_blob_proto(param_b),
    ]

    return _gen_layer(opr, "Bias", context, bias_param=param, blobs=blobs)


@_register_op(PoolingForwardOpr)
def _pooling2d(opr, context):
    assert opr.mode in [
        "MAX",
        "AVERAGE",
        "AVERAGE_COUNT_EXCLUDE_PADDING",
    ], "Pooling op doesn't support mode {}, you can implement it in _pooling2d".format(
        opr.mode
    )
    if opr.mode == "AVERAGE_COUNT_EXCLUDE_PADDING":
        logger.warning(
            "Caffe average pooling layer doesn't support 'COUNT_EXCLUDE_PADDING', you'd better set pooling mode to 'AVERAGE'"
        )
    ph, pw = opr.ph, opr.pw
    sh, sw = opr.sh, opr.sw
    kh, kw = opr.kh, opr.kw
    assert not None in opr.inp_vars[0].shape[2:4]

    ih, iw = opr.inp_vars[0].shape[2:4]
    nh = ceil((ph * 2 + ih - kh + sh) / sh)
    nw = ceil((pw * 2 + iw - kw + sw) / sw)
    if ph > 0 and (nh - 1) * sh >= ih + ph:
        nh = nh - 1
    if pw > 0 and (nw - 1) * sw >= iw + pw:
        nw = nw - 1

    param = cp.PoolingParameter(
        pool=0 if opr.mode == "MAX" else 1,
        pad_h=ph,
        pad_w=pw,
        stride_h=sh,
        stride_w=sw,
        kernel_h=kh,
        kernel_w=kw,
    )

    bottom = [context.get_blob_name(opr.inp_vars[0])]
    top = [context.set_blob_name(opr.out_vars[0], opr.name)]
    context.add_layer(
        cp.LayerParameter(
            name=opr.name, type="Pooling", bottom=bottom, top=top, pooling_param=param
        )
    )

    if (nh - 1) * sh + kh > ph * 2 + ih:
        logger.warning("Add extra 'slice layer' after Pooling Opr %s", opr.name)
        param = cp.SliceParameter(axis=2, slice_point=[nh - 1])
        bottom = top[:1]
        name = opr.name + context.gen_name
        top = [name, context.gen_name]
        context.add_layer(
            cp.LayerParameter(
                name=name, type="Slice", bottom=bottom, top=top, slice_param=param
            )
        )
        context.add_layer(silence_blob(top[1]))
    if (nw - 1) * sw + kw > pw * 2 + iw:
        logger.warning("Add extra 'slice layer' after Pooling Opr %s", opr.name)
        param = cp.SliceParameter(axis=3, slice_point=[nw - 1])
        bottom = top[:1]
        name = opr.name + context.gen_name
        top = [name, context.gen_name]
        context.add_layer(
            cp.LayerParameter(
                name=name, type="Slice", bottom=bottom, top=top, slice_param=param
            )
        )
        context.add_layer(silence_blob(top[1]))
    context.reset_blob_name(opr.out_vars[0], top[0])


@_register_op(ReshapeOpr)
def _reshape(opr, context):
    if isconst(opr.inp_vars[0]):
        if opr.out_vars[0].np_data is None:
            opr.out_vars[0].np_data = get_symvar_value(opr.out_vars[0]._var)
        return
    inp_shape = opr.input_shape
    out_shape = opr.output_shape

    if inp_shape == out_shape:
        logger.info("Input shape and output shape of Opr %s are same, skip!", opr.name)
        inp_blob = context.get_blob_name(opr.inp_vars[0])
        context.set_blob_name(opr.out_vars[0], inp_blob)
    elif (
        all([inp_shape, out_shape])
        and len(inp_shape) > len(out_shape)
        and inp_shape[0] == out_shape[0]
    ):
        d = len(inp_shape) - len(out_shape)
        tmp_shape = out_shape + (1,) * d
        bottom = [context.get_blob_name(opr.inp_vars[0])]
        top = [context.set_blob_name(opr.out_vars[0], opr.name)]
        if inp_shape != tmp_shape:
            param = cp.ReshapeParameter(shape=cp.BlobShape(dim=tmp_shape))
            tmp = [bottom[0] + "tmp"]
            context.add_layer(
                cp.LayerParameter(
                    bottom=bottom,
                    top=tmp,
                    name=opr.name + "tmp",
                    type="Reshape",
                    reshape_param=param,
                )
            )
            bottom = tmp
        param = cp.FlattenParameter(axis=len(out_shape) - 1, end_axis=-1)
        context.add_layer(
            cp.LayerParameter(
                bottom=bottom,
                top=top,
                name=opr.name,
                type="Flatten",
                flatten_param=param,
            )
        )
    else:
        logger.warning(
            "NNIE doesn't support this reshape Opr %s, NNIE reshape only support C/H/W, not N!",
            opr.name,
        )
        if out_shape is None:
            out_shape = opr.shape_param
        param = cp.ReshapeParameter(shape=cp.BlobShape(dim=out_shape))
        context.add_layer(_gen_layer(opr, "Reshape", context, reshape_param=param))


@_register_op(ConcatOpr)
def _concat(opr, context):
    param = cp.ConcatParameter(axis=opr.axis)
    context.add_layer(
        _gen_layer(opr, "Concat", context, single_input=False, concat_param=param)
    )


@_register_op(MatrixMulOpr)
def _fully_connected(opr, context):
    assert opr.inp_vars[1].np_data is not None
    param_W = opr.inp_vars[1].np_data
    assert not opr.transposeA

    if not opr.transposeB:
        param_W = param_W.T
    param = cp.InnerProductParameter(
        bias_term=False, num_output=opr.out_vars[0].shape[1]
    )
    blobs = [context.gen_blob_proto(param_W)]
    bottom = [context.get_blob_name(opr.inp_vars[0])]
    top = [context.set_blob_name(opr.out_vars[0], opr.name)]

    context.add_layer(
        cp.LayerParameter(
            name=opr.name,
            type="InnerProduct",
            bottom=bottom,
            top=top,
            inner_product_param=param,
            blobs=blobs,
        )
    )


@_register_op(BatchNormForwardOpr)
def _batchnorm(opr, context):
    inp = opr.inp_vars[0]
    scale_ = opr.scale
    bias_ = opr.bias
    mean_ = opr.mean
    var_ = opr.var
    bottom = [
        context.get_blob_name(inp),
    ]
    tmp = [bottom[0] + context.gen_name]
    top = [context.set_blob_name(opr.out_vars[opr.output_idx], opr.name)]
    bn_param = cp.BatchNormParameter(use_global_stats=True)
    bn_blobs = [
        context.gen_blob_proto(mean_),
        context.gen_blob_proto(var_),
        context.gen_blob_proto(np.array([1])),
    ]
    context.add_layer(
        cp.LayerParameter(
            bottom=bottom,
            top=tmp,
            name=opr.name + "_bn",
            type="BatchNorm",
            batch_norm_param=bn_param,
            blobs=bn_blobs,
        )
    )

    scale_param = cp.ScaleParameter(axis=1, num_axes=bias_.ndim)
    scale_param.bias_term = True
    scale_blobs = [context.gen_blob_proto(scale_), context.gen_blob_proto(bias_)]
    context.add_layer(
        cp.LayerParameter(
            bottom=tmp,
            top=top,
            name=opr.name + "_scale",
            type="Scale",
            scale_param=scale_param,
            blobs=scale_blobs,
        )
    )


@_register_op(ReduceOpr)
def _reduce(opr, context):
    assert opr.mode in [
        "SUM",
        "MAX",
    ], "Reduce op doesn't support mode {}, you can implement it in _reduce".format(
        opr.mode
    )
    if opr.mode == "SUM":
        bottom = [context.get_blob_name(opr.inp_vars[0])]
        top = [context.set_blob_name(opr.out_vars[0], opr.name)]
        context.add_layer(
            cp.LayerParameter(
                name=opr.name,
                type="Reduction",
                bottom=bottom,
                top=top,
                reduction_param=cp.ReductionParameter(
                    operation=opr.mode, axis=opr.axis
                ),
            )
        )
        param = cp.ReshapeParameter(shape=cp.BlobShape(dim=opr.out_vars[0].shape))
        bottom = top
        name = opr.name + context.gen_name
        top = [context.reset_blob_name(opr.out_vars[0], name)]
        context.add_layer(
            cp.LayerParameter(
                name=name, type="Reshape", bottom=bottom, top=top, reshape_param=param
            )
        )

    if opr.mode == "MAX":
        logger.warning("Use 'slice/concat layers' to replace operator: Reduce Max")
        bottom = [context.get_blob_name(opr.inp_vars[0])]
        name = opr.name + context.gen_name
        top = [context.gen_name for _ in range(opr.inp_vars[0].shape[opr.axis])]
        context.add_layer(
            cp.LayerParameter(
                name=name,
                type="Slice",
                bottom=bottom,
                top=top,
                slice_param=cp.SliceParameter(
                    axis=opr.axis,
                    slice_point=list(range(1, opr.inp_vars[0].shape[opr.axis])),
                ),
            )
        )
        bottom = top
        top = [context.set_blob_name(opr.out_vars[0], opr.name)]
        context.add_layer(
            cp.LayerParameter(
                name=opr.name,
                type="Eltwise",
                bottom=bottom,
                top=top,
                eltwise_param=cp.EltwiseParameter(operation="MAX"),
            )
        )


@_register_op(AxisAddRemoveOpr)
def axis_add_remove(opr, context):
    logger.warning("Use 'reshape layer' to replace operator: AxisAddRemove")
    param = cp.ReshapeParameter(shape=cp.BlobShape(dim=opr.out_vars[0].shape))
    bottom = [context.get_blob_name(opr.inp_vars[0])]
    top = [context.set_blob_name(opr.out_vars[0], opr.name)]
    context.add_layer(
        cp.LayerParameter(
            name=opr.name, type="Reshape", bottom=bottom, top=top, reshape_param=param
        )
    )
