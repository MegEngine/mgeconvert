# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=import-error
import collections
import os
from enum import IntEnum
from math import ceil
from typing import Sequence

import numpy as np
from megengine import get_logger

from ...converter_ir.ir_op import (
    AbsOpr,
    AdaptiveAvgPool2dOpr,
    AddOpr,
    AvgPool2dOpr,
    AxisAddRemoveOpr,
    BatchNormalizationOpr,
    BroadcastOpr,
    ConcatOpr,
    ConstantOpr,
    Conv2dOpr,
    Deconv2dOpr,
    ExpOpr,
    FlattenOpr,
    GetSubTensorOpr,
    GetVarShapeOpr,
    HardSigmoidOpr,
    HardSwishOpr,
    IdentityOpr,
    LeakyReluOpr,
    LinearOpr,
    LinspaceOpr,
    LogOpr,
    MatMulOpr,
    MaxOpr,
    MaxPool2dOpr,
    MinOpr,
    MulOpr,
    MultipleDeviceTensorHolderOpr,
    OpBase,
    PowOpr,
    ReduceOpr,
    Relu6Opr,
    ReluOpr,
    RepeatOpr,
    ReshapeOpr,
    SharedDeviceTensorOpr,
    SigmoidOpr,
    SiLUOpr,
    SoftmaxOpr,
    SqueezeOpr,
    SubOpr,
    TanHOpr,
    TransposeOpr,
    TrueDivOpr,
    TypeCvtOpr,
    VolatileSharedDeviceTensorOpr,
)
from ...converter_ir.ir_tensor import IRTensor
from ...frontend.mge_to_ir.mge_utils import get_symvar_value

if "USE_CAFFE_PROTO" not in os.environ:
    from .caffe_pb import caffe_pb2 as cp
else:
    from caffe.proto import caffe_pb2 as cp
logger = get_logger(__name__)
MGE2CAFFE = {}


class BackEnd(IntEnum):
    CAFFE = 1
    SNPE = 2
    TRT = 3


def isconst(x):
    return x.np_data is not None


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


def _add_input_layer(tensor, context):
    param = cp.InputParameter(shape=[cp.BlobShape(dim=tensor.shape)])
    context.add_layer(
        cp.LayerParameter(
            bottom=[],
            top=[context.set_blob_name(tensor, tensor.name)],
            name=tensor.name,
            type="Input",
            input_param=param,
        )
    )


def _gen_layer(opr, etype, context, single_input=True, **kwargs):
    bottom = (
        [context.get_blob_name(opr.inp_tensors[0])]
        if single_input
        else list(map(context.get_blob_name, opr.inp_tensors))
    )
    top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]
    return cp.LayerParameter(
        bottom=bottom, top=top, name=opr.out_tensors[0].name, type=etype, **kwargs
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


@_register_op(
    MultipleDeviceTensorHolderOpr,
    SharedDeviceTensorOpr,
    VolatileSharedDeviceTensorOpr,
    LinspaceOpr,
)
def _ignore(*_):
    pass


@_register_op(GetVarShapeOpr)
def shapeof(opr):
    out_shape = opr.out_tensors[0]
    if hasattr(out_shape, "_var"):
        out_shape.np_data = get_symvar_value(out_shape._var)
    else:
        out_shape.np_data = np.array(opr.inp_tensors[0].shape, dtype=np.int64)


@_register_op(TransposeOpr)
def _dimshfulle(opr, context):
    def swap_two_dimension(x, y, f, top, context):
        prefix = "{}_swap_{}_{}".format(opr.out_tensors[0].name, x, y)
        if opr.inp_tensors[0].shape[f[x]] > 1:
            bottom = top
            top = [context.gen_name for _ in range(opr.inp_tensors[0].shape[f[x]])]
            context.add_layer(
                cp.LayerParameter(
                    name="{}_slice_x".format(prefix),
                    type="Slice",
                    bottom=bottom,
                    top=top,
                    slice_param=cp.SliceParameter(
                        axis=x,
                        slice_point=list(range(1, opr.inp_tensors[0].shape[f[x]])),
                    ),
                )
            )
        if opr.inp_tensors[0].shape[f[y]] > 1:
            bottom = top
            top = [
                [context.gen_name for _ in range(opr.inp_tensors[0].shape[f[y]])]
                for _ in range(opr.inp_tensors[0].shape[f[x]])
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
                                slice_point=list(
                                    range(1, opr.inp_tensors[0].shape[f[y]])
                                ),
                            ),
                        ),
                        range(opr.inp_tensors[0].shape[f[x]]),
                    )
                )
            )
            bottom = top
            top = [context.gen_name for _ in range(opr.inp_tensors[0].shape[f[x]])]
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
                        range(opr.inp_tensors[0].shape[f[x]]),
                    )
                )
            )
        if opr.inp_tensors[0].shape[f[x]] > 1:
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
    assert len(l) == opr.inp_tensors[0].ndim
    nl = []
    for i, _ in enumerate(l):
        while l[i] != i:
            j = l[i]
            nl.append((i, j))
            l[i], l[j] = l[j], l[i]
    nl.reverse()

    f = list(range(len(l)))

    top = [context.get_blob_name(opr.inp_tensors[0])]
    for i in nl:
        top = swap_two_dimension(i[0], i[1], f, top, context)
    if len(opr.pattern) != opr.inp_tensors[0].ndim:
        bottom = top
        top = [context.gen_name]
        context.add_layer(
            cp.LayerParameter(
                name="{}_reshape".format(opr.out_tensors[0].name),
                type="Reshape",
                bottom=bottom,
                top=top,
                reshape_param=cp.ReshapeParameter(shape=cp.BlobShape(dim=opr.shapes)),
            )
        )
    context.set_blob_name(opr.out_tensors[0], list(top)[0])


@_register_op(GetSubTensorOpr)
def _subtensor(opr, context):
    logger.warning("Add 'slice/concat layers' in operator: Subtensor")

    top = [context.get_blob_name(opr.inp_tensors[0])]

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

    for i in range(0, len(opr.begin_params)):
        begin = opr.begin_params[i]
        end = opr.end_params[i]
        step = opr.step_params[i]
        axis = opr.axis[i]
        sl = slice(begin, end, step)

        n = opr.inp_tensors[0].shape[axis]

        slice_list = get_slice_list(n, sl, step < 0)
        if slice_list == []:
            continue
        concat_list = get_concat_list(n, axis, sl)
        if concat_list == []:
            continue

        bottom = top
        name = opr.out_tensors[0].name + next(axis_suffixes)

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

    context.set_blob_name(opr.out_tensors[0], list(top)[0])

    if len(opr.squeeze_axis):
        name = opr.out_tensors[0].name + "_reshape"
        param = cp.ReshapeParameter(shape=cp.BlobShape(dim=opr.out_tensors[0].shape))
        bottom = top
        top = [context.reset_blob_name(opr.out_tensors[0], name)]
        context.add_layer(
            cp.LayerParameter(
                name=name, type="Reshape", bottom=bottom, top=top, reshape_param=param
            )
        )
        logger.warning("Add 'reshape layers' in operator: Subtensor")


def bias_add(input, output, bias, name, context):
    param = cp.BiasParameter(axis=1, num_axes=1)
    blobs = [context.gen_blob_proto(bias)]
    bottom = [context.get_blob_name(input)]
    top = [context.set_blob_name(output, name)]
    context.add_layer(
        cp.LayerParameter(
            name=name,
            bottom=bottom,
            top=top,
            type="Bias",
            bias_param=param,
            blobs=blobs,
        )
    )


def _arith_with_const_tensor(input, const, order, opr, context):
    topB = const.np_data
    if input.ndim >= 2 and (
        topB.squeeze().shape == (input.shape[1],) or topB.squeeze().shape == (1,)
    ):
        topA = [context.get_blob_name(input)]
        topB = topB.squeeze()
        shape = topB.shape
        layer_param = cp.ScaleParameter()
    else:
        topA, topB, shape = _broadcast_for_eltwiseopr(
            opr.out_tensors[0].name, input, const, context
        )
        layer_param = cp.ScaleParameter(axis=len(shape) - topB.ndim, num_axes=topB.ndim)

    if isinstance(opr, (AddOpr, SubOpr)):
        layer_param.bias_term = True
        param_b = topB
        param_k = np.ones(shape=param_b.shape)
        if isinstance(opr, SubOpr):
            if order == 0:
                param_b = -param_b  # pylint: disable=invalid-unary-operand-type
            else:
                param_k = -param_k  # pylint: disable=invalid-unary-operand-type
        blobs = [context.gen_blob_proto(param_k), context.gen_blob_proto(param_b)]
    else:
        param_k = topB
        if isinstance(opr, TrueDivOpr):
            if order == 0:
                param_k = 1.0 / param_k
            else:
                bottom = topA
                name = opr.out_tensors[0].name + context.gen_name
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
    top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]
    context.add_layer(
        cp.LayerParameter(
            name=opr.out_tensors[0].name,
            type="Scale",
            bottom=bottom,
            top=top,
            scale_param=layer_param,
            blobs=blobs,
        )
    )


def _arith(opr, context):
    assert isinstance(opr, (AddOpr, SubOpr, TrueDivOpr, MulOpr))

    if isconst(opr.inp_tensors[0]) and isconst(opr.inp_tensors[1]):
        return
    elif isconst(opr.inp_tensors[0]) or isconst(opr.inp_tensors[1]):
        if isconst(opr.inp_tensors[0]):
            inpA = opr.inp_tensors[1]
            const = opr.inp_tensors[0]
            order = 1
        else:
            inpA = opr.inp_tensors[0]
            const = opr.inp_tensors[1]
            order = 0
        use_bias_layer = False
        bias = 0
        if isinstance(opr, AddOpr):
            bias = const.np_data.squeeze()
            if bias.ndim == 1 and inpA.ndim > 1 and bias.shape[0] == inpA.shape[1]:
                use_bias_layer = True
        if use_bias_layer:
            bias_add(inpA, opr.out_tensors[0], bias, opr.out_tensors[0].name, context)
        else:
            _arith_with_const_tensor(inpA, const, order, opr, context)
    else:
        topA, topB, _ = _broadcast_for_eltwiseopr(
            opr.out_tensors[0].name, opr.inp_tensors[0], opr.inp_tensors[1], context
        )
        if isinstance(opr, (AddOpr, SubOpr)):
            param = cp.EltwiseParameter(operation="SUM")
            if isinstance(opr, SubOpr):
                param.coeff.extend([1, -1])
        else:
            if isinstance(opr, TrueDivOpr):
                bottom = topB
                name = opr.out_tensors[0].name + context.gen_name
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
        top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]
        context.add_layer(
            cp.LayerParameter(
                name=opr.out_tensors[0].name,
                type="Eltwise",
                bottom=bottom,
                top=top,
                eltwise_param=param,
            )
        )


@_register_op(AbsOpr)
def _abs(tm_opr: OpBase, context):
    context.add_layer(_gen_layer(tm_opr, "AbsVal", context))


@_register_op(AddOpr, SubOpr, MulOpr, TrueDivOpr)
def _elemwise_arith(tm_opr: OpBase, context):
    _arith(tm_opr, context)


@_register_op(IdentityOpr)
def _indentity(tm_opr: OpBase, context):
    context.set_blob_name(
        tm_opr.out_tensors[0], context.get_blob_name(tm_opr.inp_tensors[0])
    )


@_register_op(Conv2dOpr, Deconv2dOpr)
def _convolution(opr, context):
    def expand(x):
        if isinstance(x, (list, tuple)):
            return x
        elif isinstance(x, int):
            return x, x
        else:
            raise TypeError(
                "get error type! got {} expect int or tuple[int,..]".format(type(x))
            )

    ph, pw = expand(opr.padding)
    sh, sw = expand(opr.stride)
    param_W = opr.inp_tensors[1].np_data
    kh, kw = param_W.shape[-2:]
    group = opr.groups
    dilation_h, dilation_w = expand(opr.dilation)
    assert (
        dilation_h == dilation_w
    ), "caffe accept one dilation, so dilation_h and dilation_w must equal"
    param_W = param_W.reshape((-1,) + param_W.shape[-3:])
    bias_term = len(opr.inp_tensors) > 2
    blobs = [
        context.gen_blob_proto(param_W),
    ]
    param = cp.ConvolutionParameter(
        num_output=opr.out_tensors[0].shape[1],
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
    top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]
    if isinstance(opr, Deconv2dOpr):
        layer_type = "Deconvolution"
        context.set_blob_name(opr.inp_tensors[1], opr.inp_tensors[0].name)
        bottom = [context.get_blob_name(opr.inp_tensors[1])]
    else:
        layer_type = "Convolution"
        bottom = [context.get_blob_name(opr.inp_tensors[0])]

    if len(opr.inp_tensors) > 2:
        blobs.append(context.gen_blob_proto(opr.inp_tensors[2].np_data.reshape(-1,)))
        assert bias_term == True
    else:
        assert bias_term == False

    context.add_layer(
        cp.LayerParameter(
            bottom=bottom,
            top=top,
            name=opr.out_tensors[0].name,
            type=layer_type,
            blobs=blobs,
            convolution_param=param,
        )
    )


@_register_op(AvgPool2dOpr, MaxPool2dOpr, AdaptiveAvgPool2dOpr)
def _pooling2d(opr, context):
    # assert opr.mode in [
    # 	  "MAX",
    # 	  "AVERAGE",
    # 	  "AVERAGE_COUNT_EXCLUDE_PADDING",
    # ], "Pooling op doesn't support mode {}, you can implement it in _pooling2d".format(
    # 	  opr.mode
    # )
    def _unexpand(x):
        if isinstance(x, Sequence):
            return x[0], x[1]
        elif isinstance(x, int):
            return x, x
        else:
            raise TypeError(
                "get error type! got {} expect int or tuple[int,..]".format(type(x))
            )

    if not isinstance(opr, AdaptiveAvgPool2dOpr):
        if opr.mode == "AVERAGE_COUNT_EXCLUDE_PADDING":
            logger.warning(
                "Caffe average pooling layer doesn't support 'COUNT_EXCLUDE_PADDING', you'd better set pooling mode to 'AVERAGE'"
            )

        ph, pw = _unexpand(opr.padding)
        sh, sw = _unexpand(opr.stride)
        kh, kw = _unexpand(opr.kernel_size)
        assert not None in opr.inp_tensors[0].shape[2:4]

        ih, iw = opr.inp_tensors[0].shape[2:4]
        nh = ceil((ph * 2 + ih - kh + sh) / sh)
        nw = ceil((pw * 2 + iw - kw + sw) / sw)
        if ph > 0 and (nh - 1) * sh >= ih + ph:
            nh = nh - 1
        if pw > 0 and (nw - 1) * sw >= iw + pw:
            nw = nw - 1
    elif isinstance(opr, AdaptiveAvgPool2dOpr):
        oh, ow = _unexpand(opr.out_shape)
        ih, iw = list(opr.inp_tensors[0].shape)[-2:]
        ph, pw = 0, 0
        sh, sw = ih // oh, iw // ow
        kh, kw = ih - (oh - 1) * sh, iw - (ow - 1) * sw

    if hasattr(opr, "mode"):
        pool_mode = 0 if opr.mode == "MAX" else 1
    else:
        pool_mode = 1

    param = cp.PoolingParameter(
        pool=pool_mode,
        pad_h=ph,
        pad_w=pw,
        stride_h=sh,
        stride_w=sw,
        kernel_h=kh,
        kernel_w=kw,
    )

    bottom = [context.get_blob_name(opr.inp_tensors[0])]
    top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]
    context.add_layer(
        cp.LayerParameter(
            name=opr.out_tensors[0].name,
            type="Pooling",
            bottom=bottom,
            top=top,
            pooling_param=param,
        )
    )

    if not isinstance(opr, AdaptiveAvgPool2dOpr):
        if (nh - 1) * sh + kh > ph * 2 + ih:
            logger.warning(
                "Add extra 'slice layer' after Pooling Opr %s", opr.out_tensors[0].name
            )
            param = cp.SliceParameter(axis=2, slice_point=[nh - 1])
            bottom = top[:1]
            name = opr.out_tensors[0].name + context.gen_name
            top = [name, context.gen_name]
            context.add_layer(
                cp.LayerParameter(
                    name=name, type="Slice", bottom=bottom, top=top, slice_param=param
                )
            )
            context.add_layer(silence_blob(top[1]))
        if (nw - 1) * sw + kw > pw * 2 + iw:
            logger.warning(
                "Add extra 'slice layer' after Pooling Opr %s", opr.out_tensors[0].name
            )
            param = cp.SliceParameter(axis=3, slice_point=[nw - 1])
            bottom = top[:1]
            name = opr.out_tensors[0].name + context.gen_name
            top = [name, context.gen_name]
            context.add_layer(
                cp.LayerParameter(
                    name=name, type="Slice", bottom=bottom, top=top, slice_param=param
                )
            )
            context.add_layer(silence_blob(top[1]))
    context.reset_blob_name(opr.out_tensors[0], top[0])


@_register_op(ConcatOpr)
def _concat(opr, context):
    param = cp.ConcatParameter(axis=opr.axis)
    context.add_layer(
        _gen_layer(opr, "Concat", context, single_input=False, concat_param=param)
    )


@_register_op(SigmoidOpr, TanHOpr, LogOpr, ExpOpr)
def _elemwise_single_layer(opr, context):
    context.add_layer(_gen_layer(opr, opr.name, context))


@_register_op(MaxOpr)
def _elemwise_max(opr, context):
    param = cp.EltwiseParameter(operation="MAX")
    assert (
        opr.inp_tensors[0].np_data is None and opr.inp_tensors[1].np_data is None
    ), "Caffe doesn't support elemwise MAX(tensor, const)"
    context.add_layer(
        _gen_layer(opr, "Eltwise", context, single_input=False, eltwise_param=param)
    )


@_register_op(MinOpr)
def _elemwise_min(opr, context):
    param = cp.EltwiseParameter(operation="MIN")
    assert (
        opr.inp_tensors[0].np_data is None and opr.inp_tensors[1].np_data is None
    ), "Caffe doesn't support elemwise MIN(tensor, const)"
    context.add_layer(
        _gen_layer(opr, "Eltwise", context, single_input=False, eltwise_param=param)
    )


@_register_op(PowOpr)
def _pow(opr, context):
    power = opr.inp_tensors[1].np_data
    assert power.size == 1  # must with one power number
    power_param = cp.PowerParameter(scale=1, shift=0, power=power.item())
    context.add_layer(_gen_layer(opr, "Power", context, power_param=power_param))


@_register_op(ReduceOpr)
def _reduce(opr, context):
    assert opr.mode in [
        "SUM",
        "MAX",
        "SUM_SQR",
        "MEAN",
    ], "Reduce op doesn't support mode {}, you can implement it in _reduce".format(
        opr.mode
    )
    if opr.mode in ("SUM", "SUM_SQR", "MEAN"):
        mode = opr.mode
        if opr.mode == "SUM_SQR":
            mode = "SUMSQ"
        bottom = [context.get_blob_name(opr.inp_tensors[0])]
        top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]
        context.add_layer(
            cp.LayerParameter(
                name=opr.out_tensors[0].name,
                type="Reduction",
                bottom=bottom,
                top=top,
                reduction_param=cp.ReductionParameter(operation=mode, axis=opr.axis),
            )
        )
        param = cp.ReshapeParameter(shape=cp.BlobShape(dim=opr.out_tensors[0].shape))
        bottom = top
        name = opr.out_tensors[0].name + context.gen_name
        top = [context.reset_blob_name(opr.out_tensors[0], name)]
        context.add_layer(
            cp.LayerParameter(
                name=name, type="Reshape", bottom=bottom, top=top, reshape_param=param
            )
        )

    if opr.mode == "MAX":
        logger.warning("Use 'slice/concat layers' to replace operator: Reduce Max")
        bottom = [context.get_blob_name(opr.inp_tensors[0])]
        name = opr.out_tensors[0].name + context.gen_name
        top = [context.gen_name for _ in range(opr.inp_tensors[0].shape[opr.axis])]
        context.add_layer(
            cp.LayerParameter(
                name=name,
                type="Slice",
                bottom=bottom,
                top=top,
                slice_param=cp.SliceParameter(
                    axis=opr.axis,
                    slice_point=list(range(1, opr.inp_tensors[0].shape[opr.axis])),
                ),
            )
        )
        bottom = top
        top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]
        context.add_layer(
            cp.LayerParameter(
                name=opr.out_tensors[0].name,
                type="Eltwise",
                bottom=bottom,
                top=top,
                eltwise_param=cp.EltwiseParameter(operation="MAX"),
            )
        )

        if not opr.keep_dims:
            param = cp.ReshapeParameter(
                shape=cp.BlobShape(dim=opr.out_tensors[0].shape)
            )
            bottom = top
            name = opr.out_tensors[0].name + context.gen_name
            top = [context.reset_blob_name(opr.out_tensors[0], name)]
            context.add_layer(
                cp.LayerParameter(
                    name=name,
                    type="Reshape",
                    bottom=bottom,
                    top=top,
                    reshape_param=param,
                )
            )


@_register_op(ReluOpr)
def _relu(tm_opr: OpBase, context):
    context.add_layer(_gen_layer(tm_opr, "ReLU", context))


@_register_op(SoftmaxOpr)
def _softmax(tm_opr: OpBase, context):
    context.add_layer(_gen_layer(tm_opr, "Softmax", context))


@_register_op(ReshapeOpr)
def _reshape(opr, context):
    if isconst(opr.inp_tensors[0]):
        # delete set out_tensor np data
        if opr.out_tensors[0].np_data is None:
            if hasattr(opr.out_tensors[0], "_var"):
                opr.out_tensors[0].np_data = get_symvar_value(opr.out_tensors[0]._var)
            else:
                opr.out_tensors[0].np_data = opr.inp_tensors[0].np_data.reshape(
                    opr.out_shape
                )
        return
    inp_shape = opr.inp_tensors[0].shape
    out_shape = tuple(opr.out_shape)

    if inp_shape == out_shape:
        logger.info(
            "Input shape and output shape of Opr %s are same, skip!",
            opr.out_tensors[0].name,
        )
        inp_blob = context.get_blob_name(opr.inp_tensors[0])
        context.set_blob_name(opr.out_tensors[0], inp_blob)
    elif (
        all([inp_shape, out_shape])
        and len(inp_shape) >= len(out_shape)
        and inp_shape[0] == out_shape[0]
    ):
        d = len(inp_shape) - len(out_shape)
        tmp_shape = out_shape + (1,) * d
        bottom = [context.get_blob_name(opr.inp_tensors[0])]
        top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]
        if context.convert_backend == BackEnd.CAFFE:
            if inp_shape != tmp_shape:
                logger.warning(
                    "trt don't support flatten after reshape, but caffe support! please use BackEnd.TRT in tracedmodule_to_caffe by pass convert_backend:BackEnd=BackEnd.TRT"
                )
                param = cp.ReshapeParameter(shape=cp.BlobShape(dim=tmp_shape))
                tmp = [bottom[0] + "tmp"]
                context.add_layer(
                    cp.LayerParameter(
                        bottom=bottom,
                        top=tmp,
                        name=opr.out_tensors[0].name + "tmp",
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
                    name=opr.out_tensors[0].name,
                    type="Flatten",
                    flatten_param=param,
                )
            )
        elif context.convert_backend == BackEnd.TRT:
            if inp_shape != tmp_shape:
                param = cp.ReshapeParameter(shape=cp.BlobShape(dim=tmp_shape))
                context.add_layer(
                    cp.LayerParameter(
                        bottom=bottom,
                        top=top,
                        name=opr.out_tensors[0].name,
                        type="Reshape",
                        reshape_param=param,
                    )
                )
    else:
        logger.warning(
            "NNIE doesn't support this reshape Opr %s, inp_shape %s, out_shape %s, NNIE reshape only support C/H/W, not N!",
            opr.out_tensors[0].name,
            inp_shape,
            out_shape,
        )
        if out_shape is None:
            out_shape = opr.shape_param
        param = cp.ReshapeParameter(shape=cp.BlobShape(dim=out_shape))
        context.add_layer(_gen_layer(opr, "Reshape", context, reshape_param=param))


@_register_op(ConstantOpr)
def _constant(*_):
    pass


@_register_op(FlattenOpr)
def _flatten_shape(opr, context):
    bottom = [context.get_blob_name(opr.inp_tensors[0])]
    top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]
    param = cp.FlattenParameter(axis=opr.start_axis, end_axis=opr.end_axis)
    context.add_layer(
        cp.LayerParameter(
            bottom=bottom,
            top=top,
            name=opr.out_tensors[0].name,
            type="Flatten",
            flatten_param=param,
        )
    )


@_register_op(MatMulOpr, LinearOpr)
def _fully_connected(opr, context):
    assert opr.inp_tensors[1].np_data is not None
    param_W = opr.inp_tensors[1].np_data
    assert not opr.transpose_a

    if not opr.transpose_b:
        param_W = param_W.T

    blobs = [context.gen_blob_proto(param_W)]
    bias_term = False

    if isinstance(opr, LinearOpr) and opr.has_bias:
        bias_term = True
        blobs.append(context.gen_blob_proto(opr.inp_tensors[2].np_data.reshape(-1,)))

    param = cp.InnerProductParameter(
        bias_term=bias_term, num_output=opr.out_tensors[0].shape[1]
    )
    bottom = [context.get_blob_name(opr.inp_tensors[0])]
    top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]

    context.add_layer(
        cp.LayerParameter(
            name=opr.out_tensors[0].name,
            type="InnerProduct",
            bottom=bottom,
            top=top,
            inner_product_param=param,
            blobs=blobs,
        )
    )


@_register_op(SqueezeOpr)
def _squeeze(opr, context):
    logger.warning("Use 'reshape layer' to replace operator: AxisAddRemove")
    param = cp.ReshapeParameter(shape=cp.BlobShape(dim=opr.out_tensors[0].shape))
    bottom = [context.get_blob_name(opr.inp_tensors[0])]
    top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]
    context.add_layer(
        cp.LayerParameter(
            name=opr.out_tensors[0].name,
            type="Reshape",
            bottom=bottom,
            top=top,
            reshape_param=param,
        )
    )


@_register_op(AxisAddRemoveOpr)
def _axis_add_remove(opr, context):
    logger.warning("Use 'reshape layer' to replace operator: AxisAddRemove")
    param = cp.ReshapeParameter(shape=cp.BlobShape(dim=opr.out_tensors[0].shape))
    bottom = [context.get_blob_name(opr.inp_tensors[0])]
    top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]
    context.add_layer(
        cp.LayerParameter(
            name=opr.name, type="Reshape", bottom=bottom, top=top, reshape_param=param
        )
    )


@_register_op(LeakyReluOpr)
def _leaky_relu(opr, context):
    param = cp.ReLUParameter(negative_slope=opr.negative_slope)
    context.add_layer(_gen_layer(opr, "ReLU", context, relu_param=param))


@_register_op(Relu6Opr)
def relu6(opr, context):
    param = cp.ClipParameter(min=0.0, max=6.0)
    context.add_layer(_gen_layer(opr, "Clip", context, clip_param=param))


def add_3_relu6_div_6(opr, context, out):
    inp = opr.inp_tensors[0]
    const_3 = opr.inp_tensors[1]
    const_6 = opr.inp_tensors[2]
    fake_add_3_out = opr.inp_tensors[3]
    fake_relu6_out = opr.inp_tensors[4]
    split_add_op = AddOpr()
    split_add_op.add_inp_tensors(inp)
    split_add_op.add_inp_tensors(const_3)
    split_add_op.add_out_tensors(fake_add_3_out)
    _arith(split_add_op, context)
    relu6_op = Relu6Opr()
    relu6_op.add_inp_tensors(fake_add_3_out)
    relu6_op.add_out_tensors(fake_relu6_out)
    relu6(relu6_op, context)
    true_div_op = TrueDivOpr()
    true_div_op.add_inp_tensors(fake_relu6_out)
    true_div_op.add_inp_tensors(const_6)
    true_div_op.add_out_tensors(out)
    _arith(true_div_op, context)


@_register_op(HardSigmoidOpr)
def hsigmoid(opr, context):
    add_3_relu6_div_6(opr, context, opr.out_tensors[0])


@_register_op(HardSwishOpr)
def hswish(opr, context):
    inp = opr.inp_tensors[0]
    fake_div6_out = opr.inp_tensors[5]
    add_3_relu6_div_6(opr, context, fake_div6_out)
    mul_op = MulOpr()
    mul_op.add_inp_tensors(inp)
    mul_op.add_inp_tensors(fake_div6_out)
    mul_op.add_out_tensors(opr.out_tensors[0])
    _arith(mul_op, context)


@_register_op(SiLUOpr)
def silu(opr, context):
    inp = opr.inp_tensors[0]
    sigmoid_op = SigmoidOpr()
    sigmoid_op.add_inp_tensors(inp)
    fake_sigmoid_out = IRTensor(
        inp.name + "_sigmoid_out",
        inp.shape,
        inp.dtype,
        scale=inp.scale,
        zero_point=inp.zero_point,
        q_type=inp.q_dtype,
    )
    context.update_quantize_dict(fake_sigmoid_out)
    sigmoid_op.add_out_tensors(fake_sigmoid_out)
    context.add_layer(_gen_layer(sigmoid_op, sigmoid_op.name, context))
    mul_op = MulOpr()
    mul_op.add_inp_tensors(inp)
    mul_op.add_inp_tensors(fake_sigmoid_out)
    mul_op.add_out_tensors(opr.out_tensors[0])
    _arith(mul_op, context)


@_register_op(TypeCvtOpr)
def _typecvt(opr, context):
    context.set_blob_name(opr.out_tensors[0], context.get_blob_name(opr.inp_tensors[0]))


@_register_op(BroadcastOpr)
def _broadcast(opr, context):
    input = opr.inp_tensors[0]
    inp_ndim = input.ndim
    a_shape = input.shape
    b_shape = opr.inp_tensors[1].np_data
    b_ndim = len(b_shape)
    assert inp_ndim <= b_ndim
    bottom = [context.get_blob_name(input)]
    if inp_ndim < b_ndim:
        a_shape = (1,) * (b_ndim - inp_ndim) + a_shape
        inp_ndim = b_ndim
        param = cp.ReshapeParameter(shape=cp.BlobShape(dim=a_shape))
        top = [bottom[0] + "_reshape"]
        context.add_layer(
            cp.LayerParameter(
                bottom=bottom,
                top=top,
                name=opr.out_tensors[0].name + context.gen_name,
                type="Reshape",
                reshape_param=param,
            )
        )
        bottom = top
    for i in range(b_ndim):
        shpA, shpB = a_shape[i], b_shape[i]
        assert shpA in (shpB, 1)
        name = opr.out_tensors[0].name + context.gen_name
        top = [name]
        context.add_layer(
            cp.LayerParameter(
                name=name,
                type="Tile",
                bottom=bottom,
                top=top,
                tile_param=cp.TileParameter(axis=i, tiles=shpB),
            )
        )
        bottom = top
    context.set_blob_name(opr.out_tensors[0], bottom[0])


@_register_op(BatchNormalizationOpr)
def _batchnorm(opr, context):
    inp = opr.inp_tensors[0]
    scale_ = opr.inp_tensors[1].np_data.squeeze()
    bias_ = opr.inp_tensors[2].np_data.squeeze()
    mean_ = opr.inp_tensors[3].np_data.squeeze()
    var_ = opr.inp_tensors[4].np_data.squeeze()
    bottom = [
        context.get_blob_name(inp),
    ]
    tmp = [bottom[0] + context.gen_name]
    top = [
        context.set_blob_name(
            opr.out_tensors[opr.output_idx], opr.out_tensors[opr.output_idx].name
        )
    ]
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
            name=opr.out_tensors[opr.output_idx].name + "_bn",
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
            name=opr.out_tensors[opr.output_idx].name + "_scale",
            type="Scale",
            scale_param=scale_param,
            blobs=scale_blobs,
        )
    )


@_register_op(RepeatOpr)
def _fake_repeat(opr, context):
    unsqueeze_shape = list(opr.inp_tensors[0].shape)
    unsqueeze_shape.insert(opr.axis + 1, 1)
    fake_unsqueeze_out = IRTensor(
        opr.inp_tensors[0].name + "_unsqueeze",
        unsqueeze_shape,
        opr.inp_tensors[0].dtype,
        q_type=opr.inp_tensors[0].q_dtype,
        scale=opr.inp_tensors[0].scale,
        zero_point=opr.inp_tensors[0].zero_point,
    )
    context.update_quantize_dict(fake_unsqueeze_out)
    param = cp.ReshapeParameter(shape=cp.BlobShape(dim=unsqueeze_shape))
    bottom = [context.get_blob_name(opr.inp_tensors[0])]
    top = [context.set_blob_name(fake_unsqueeze_out, fake_unsqueeze_out.name)]
    context.add_layer(
        cp.LayerParameter(
            name=fake_unsqueeze_out.name,
            type="Reshape",
            bottom=bottom,
            top=top,
            reshape_param=param,
        )
    )
    param = cp.TileParameter(axis=opr.axis + 1, tiles=opr.repeats)
    unsqueeze_shape[opr.axis + 1] = unsqueeze_shape[opr.axis + 1] * opr.repeats
    fake_tile = IRTensor(
        opr.inp_tensors[0].name + "_unsqueeze_tile",
        unsqueeze_shape,
        opr.inp_tensors[0].dtype,
        q_type=opr.inp_tensors[0].q_dtype,
        scale=opr.inp_tensors[0].scale,
        zero_point=opr.inp_tensors[0].zero_point,
    )
    context.update_quantize_dict(fake_tile)
    bottom = top
    top = [context.set_blob_name(fake_tile, fake_tile.name)]
    context.add_layer(
        cp.LayerParameter(
            name=fake_tile.name, type="Tile", bottom=bottom, top=top, tile_param=param
        )
    )
    param = cp.ReshapeParameter(shape=cp.BlobShape(dim=opr.out_tensors[0].shape))
    bottom = top
    top = [context.set_blob_name(opr.out_tensors[0], opr.out_tensors[0].name)]
    context.add_layer(
        cp.LayerParameter(
            name=opr.out_tensors[0].name,
            type="Reshape",
            bottom=bottom,
            top=top,
            reshape_param=param,
        )
    )
