# pylint: disable=import-error,no-name-in-module,no-member,unused-argument
import builtins
import operator
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.fx
import torch.nn.functional

from ...converter_ir.ir_op import (
    AbsOpr,
    AddOpr,
    AvgPool2dOpr,
    BatchNormalizationOpr,
    CeilOpr,
    ConcatOpr,
    ConstantOpr,
    Conv2dOpr,
    ConvRelu2dOpr,
    Deconv2dOpr,
    ExpOpr,
    FlattenOpr,
    FloorDivOpr,
    FloorOpr,
    GetSubTensorOpr,
    GetVarShapeOpr,
    LeakyReluOpr,
    LinearOpr,
    LogOpr,
    MaxOpr,
    MaxPool2dOpr,
    MulOpr,
    PadOpr,
    PixelShuffle,
    PowOpr,
    ReduceOpr,
    Relu6Opr,
    ReluOpr,
    RepeatOpr,
    ReshapeOpr,
    ResizeOpr,
    SigmoidOpr,
    SoftmaxOpr,
    SqueezeOpr,
    SubOpr,
    TransposeOpr,
    TrueDivOpr,
    TypeCvtOpr,
)

MGE2FX = {}

DTYPE_MAPPING = {
    np.float32: torch.float32,
    np.int32: torch.int32,
    np.float16: torch.float16,
    np.int8: torch.int8,
    np.int32: torch.int32,
}


def _register_op(*oprs):
    def callback(impl: Callable):
        for op in oprs:
            MGE2FX[op] = impl
        return impl

    return callback


def _fallback_name(func):
    def replaced_fuc(opr, graph, load_arg, name: Optional[str] = None):
        if not name:
            name = opr.out_tensors[0].name
        return func(opr, graph, load_arg, name=name)

    return replaced_fuc


def make_func_node(
    *, graph: torch.fx.Graph, name: str, func: Callable, args: Tuple = (), **kwargs,
):
    node = graph.create_node(
        op="call_function", target=func, args=args, kwargs=kwargs, name=name
    )
    return node


def make_method_node(
    *, graph: torch.fx.Graph, name: str, func: str, args: Tuple = (), **kwargs,
):
    node = graph.create_node(
        op="call_method", target=func, args=args, kwargs=kwargs, name=name
    )
    return node


@_register_op(Conv2dOpr)
@_fallback_name
def conv2d(opr: Conv2dOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    with_act = opr.activation == "RELU"

    weight_node = load_arg(opr.inp_tensors[1])
    weight_shape = opr.inp_tensors[1].shape
    if opr.groups and opr.groups > 1:
        weight_node = make_func_node(
            graph=graph,
            name=opr.inp_tensors[1].name + "_reshaped",
            func=torch.reshape,
            args=(weight_node,),
            shape=(
                weight_shape[0] * weight_shape[1],
                weight_shape[2],
                weight_shape[3],
                weight_shape[4],
            ),
        )

    node = make_func_node(
        graph=graph,
        name=name + ("_before_act" if with_act else ""),
        func=torch.nn.functional.conv2d,
        input=load_arg(opr.inp_tensors[0]),
        weight=weight_node,
        bias=load_arg(opr.inp_tensors[2], enable_quant=False)
        if len(opr.inp_tensors) > 2
        else None,
        stride=opr.stride,
        padding=opr.padding,
        dilation=opr.dilation,
        groups=opr.groups,
    )
    if opr.activation == "RELU":
        node = relu(opr, graph, lambda x: node, name)[0]
    return [node]


@_register_op(BatchNormalizationOpr)
@_fallback_name
def batchnorm(
    opr: BatchNormalizationOpr, graph: torch.fx.Graph, load_arg: Callable, name: str
):
    # todo: fix len(opr.inp_tensors) != 5
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.nn.functional.batch_norm,
        input=load_arg(opr.inp_tensors[0]),
        running_mean=load_arg(opr.inp_tensors[3]),
        running_var=load_arg(opr.inp_tensors[4]),
        weight=load_arg(opr.inp_tensors[1]),
        bias=load_arg(opr.inp_tensors[2]),
        training=False,
        momentum=opr.momentum,
        eps=opr.eps,
    )
    return [node]


@_register_op(ConcatOpr)
@_fallback_name
def concat(opr: ConcatOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.concat,
        tensors=[load_arg(x) for x in opr.inp_tensors],
        dim=opr.axis,
    )
    return [node]


@_register_op(AddOpr)
@_fallback_name
def add(opr: AddOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=operator.add,
        args=(load_arg(opr.inp_tensors[0]), load_arg(opr.inp_tensors[1])),
    )
    return [node]


@_register_op(AvgPool2dOpr)
@_fallback_name
def avgpool(opr: AvgPool2dOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.nn.functional.avg_pool2d,
        input=load_arg(opr.inp_tensors[0]),
        kernel_size=opr.kernel_size,
        stride=opr.stride,
        padding=opr.padding,
        ceil_mode=opr.ceil_mode != 0,
    )
    return [node]


@_register_op(MaxPool2dOpr)
@_fallback_name
def maxpool(opr: MaxPool2dOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.nn.functional.max_pool2d,
        input=load_arg(opr.inp_tensors[0]),
        kernel_size=opr.kernel_size,
        stride=opr.stride,
        padding=opr.padding,
        ceil_mode=opr.ceil_mode != 0,
    )
    return [node]


@_register_op(PadOpr)
@_fallback_name
def pad(opr: PadOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.nn.functional.pad,
        input=load_arg(opr.inp_tensors[0]),
        pad=opr.pad_width,
        mode=opr.mode,
        value=opr.pad_val,
    )
    return [node]


@_register_op(ReshapeOpr)
@_fallback_name
def reshape(opr: ReshapeOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.reshape,
        input=load_arg(opr.inp_tensors[0]),
        shape=opr.out_shape,
    )
    return [node]


@_register_op(LinearOpr)
@_fallback_name
def linear(opr: LinearOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.nn.functional.linear,
        input=load_arg(opr.inp_tensors[0]),
        weight=load_arg(opr.inp_tensors[1]),
        bias=load_arg(opr.inp_tensors[2], enable_quant=False) if opr.has_bias else None,
        # todo: transpose_b
    )
    return [node]


@_register_op(ReluOpr)
@_fallback_name
def relu(opr, graph: torch.fx.Graph, load_arg: Callable, name):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.nn.functional.relu,
        input=load_arg(opr.inp_tensors[0]),
    )
    return [node]


@_register_op(ConvRelu2dOpr)
@_fallback_name
def convrelu2d(
    opr: ConvRelu2dOpr, graph: torch.fx.Graph, load_arg: Callable, name: str
):
    node = conv2d(opr, graph, load_arg, name + "_before_act")[0]
    node = relu(opr, graph, lambda x: node, name)[0]
    return [node]


@_register_op(Deconv2dOpr)
@_fallback_name
def deconv2d(opr: Deconv2dOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    with_act = opr.activation == "RELU"
    node = make_func_node(
        graph=graph,
        name=name + ("_before_act" if with_act else ""),
        func=torch.nn.functional.conv_transpose2d,
        input=load_arg(opr.inp_tensors[0]),
        weight=load_arg(opr.inp_tensors[1]),
        bias=load_arg(opr.inp_tensors[2], enable_quant=False)
        if len(opr.inp_tensors) > 2
        else None,
        stride=opr.stride,
        padding=opr.padding,
        dilation=opr.dilation,
        groups=opr.groups,
    )
    if opr.activation == "RELU":
        node = relu(opr, graph, lambda x: node, name)[0]
    return [node]


@_register_op(MulOpr)
@_fallback_name
def mul(opr: MulOpr, graph: torch.fx.Graph, load_arg: Callable, name):
    node = make_func_node(
        graph=graph,
        name=name,
        func=operator.mul,
        args=(load_arg(opr.inp_tensors[0]), load_arg(opr.inp_tensors[1])),
    )
    return [node]


@_register_op(SubOpr)
@_fallback_name
def sub(opr: SubOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=operator.sub,
        args=(load_arg(opr.inp_tensors[0]), load_arg(opr.inp_tensors[1])),
    )
    return [node]


@_register_op(AbsOpr)
@_fallback_name
def abs(opr: AbsOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph, name=name, func=operator.abs, args=(load_arg(opr.inp_tensors[0]),),
    )
    return [node]


@_register_op(LeakyReluOpr)
@_fallback_name
def leak_relu(opr: LeakyReluOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.nn.functional.leaky_relu,
        args=(load_arg(opr.inp_tensors[0]),),
        negative_slope=opr.negative_slope,
    )
    return [node]


@_register_op(SqueezeOpr)
@_fallback_name
def squeeze(opr: SqueezeOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    remove_axis = []
    for axis in opr.squeeze_dims:
        remove_axis.append(axis)
    if len(remove_axis) == 0:
        node = make_func_node(
            graph=graph,
            name=name,
            func=torch.squeeze,
            args=(load_arg(opr.inp_tensors[0]),),
        )
    else:
        node = make_func_node(
            graph=graph,
            name=name,
            func=torch.squeeze,
            args=(load_arg(opr.inp_tensors[0]),),
            dim=remove_axis[0],
        )
        for dim in remove_axis[1:]:
            node = make_func_node(
                graph=graph, name=name, func=torch.squeeze, args=node, dim=dim,
            )
    return [node]


@_register_op(PowOpr)
@_fallback_name
def pow(opr: PowOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.pow,
        args=(load_arg(opr.inp_tensors[0]),),
        exponent=2,
    )
    return [node]


@_register_op(Relu6Opr)
@_fallback_name
def relu6(opr: Relu6Opr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.nn.functional.relu6,
        args=(load_arg(opr.inp_tensors[0]),),
    )
    return [node]


@_register_op(SigmoidOpr)
@_fallback_name
def sigmoid(opr: SigmoidOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.nn.functional.sigmoid,
        args=(load_arg(opr.inp_tensors[0]),),
    )
    return [node]


@_register_op(SoftmaxOpr)
@_fallback_name
def softmax(opr: SoftmaxOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.nn.functional.softmax,
        args=(load_arg(opr.inp_tensors[0]),),
        dim=opr.axis,
    )
    return [node]


@_register_op(TransposeOpr)
@_fallback_name
def permute(opr: TransposeOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.permute,
        args=(load_arg(opr.inp_tensors[0]),),
        dims=opr.pattern,
    )
    return [node]


@_register_op(TypeCvtOpr)
@_fallback_name
def astype(opr: TypeCvtOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_method_node(
        graph=graph,
        name=name,
        func="type",
        args=(load_arg(opr.inp_tensors[0]), DTYPE_MAPPING[opr.out_dtype]),
    )
    return [node]


@_register_op(GetVarShapeOpr)
@_fallback_name
def shape(opr: GetVarShapeOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=builtins.getattr,
        args=(load_arg(opr.inp_tensors[0]), "shape"),
    )
    return [node]


@_register_op(GetSubTensorOpr)
@_fallback_name
def getitem(opr: GetSubTensorOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    indexs: List = [slice(None, None, None) for i in range(opr.inp_tensors[0].ndim)]
    for idx, axis in enumerate(opr.axis):
        if axis in opr.squeeze_axis:
            indexs[axis] = opr.begin_params[idx]
        else:
            indexs[axis] = slice(
                opr.begin_params[idx], opr.end_params[idx], opr.step_params[idx]
            )
    indexs_args: Tuple = tuple(indexs)
    if len(indexs) == 1:
        indexs_args = indexs_args[0]
    node = make_func_node(
        graph=graph,
        name=name,
        func=operator.getitem,
        args=(load_arg(opr.inp_tensors[0]), indexs_args),
    )
    return [node]


@_register_op(FloorDivOpr)
@_fallback_name
def floordiv(opr: FloorDivOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):

    node = make_func_node(
        graph=graph,
        name=name,
        func=operator.floordiv,
        args=(load_arg(opr.inp_tensors[0]), load_arg(opr.inp_tensors[1])),
    )
    return [node]


@_register_op(TrueDivOpr)
@_fallback_name
def truediv(opr: TrueDivOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=operator.truediv,
        args=(load_arg(opr.inp_tensors[0]), load_arg(opr.inp_tensors[1])),
    )
    return [node]


@_register_op(ConstantOpr)
@_fallback_name
def const(opr: ConstantOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    return [None]


@_register_op(ResizeOpr)
@_fallback_name
def resize(opr: ResizeOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.nn.functional.interpolate,
        args=(load_arg(opr.inp_tensors[0]),),
        size=opr.out_size,
        scale_factor=opr.scale_factor,
        align_corners=opr.align_corners,
        mode=opr.mode,
    )
    return [node]


@_register_op(ExpOpr)
@_fallback_name
def exp(opr: ExpOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph, name=name, func=torch.exp, args=(load_arg(opr.inp_tensors[0]),),
    )
    return [node]


@_register_op(LogOpr)
@_fallback_name
def log(opr: LogOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph, name=name, func=torch.log, args=(load_arg(opr.inp_tensors[0]),),
    )
    return [node]


@_register_op(CeilOpr)
@_fallback_name
def ceil(opr: CeilOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph, name=name, func=torch.ceil, args=(load_arg(opr.inp_tensors[0]),),
    )
    return [node]


@_register_op(FloorOpr)
@_fallback_name
def floor(opr: FloorOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph, name=name, func=torch.floor, args=(load_arg(opr.inp_tensors[0]),),
    )
    return [node]


@_register_op(MaxOpr)
@_fallback_name
def maximum(opr: MaxOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.maximum,
        args=(load_arg(opr.inp_tensors[0]), load_arg(opr.inp_tensors[1])),
    )
    return [node]


@_register_op(ReduceOpr)
@_fallback_name
def reduce(opr: ReduceOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    func: Optional[Callable] = None
    if opr.mode == "SUM":
        func = torch.sum
    if opr.mode == "MAX":
        func = torch.max
    if opr.mode == "MEAN":
        func = torch.mean
    if opr.mode == "MIN":
        func = torch.min
    assert func, "only support reduce_max and reduce_sum"
    node = make_func_node(
        graph=graph,
        name=name,
        func=func,
        args=(load_arg(opr.inp_tensors[0]),),
        dim=opr.axis,
        keepdim=opr.keep_dims,
    )
    if opr.mode in ["MIN", "MAX"]:
        node = make_func_node(
            graph=graph,
            name=name + "_values",
            func=builtins.getattr,
            args=(node, "values"),
        )
    return [node]


# @_register_op(BroadcastOpr)
# @_fallback_name
# def broadcast(opr: BroadcastOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
#     node = make_func_node(
#         graph=graph,
#         name=name,
#         func=torch.broadcast_to,
#         args=(load_arg(opr.inp_tensors[0]),),
#         size=(load_arg(opr.inp_tensors[1]),),
#     )
#     return [node]


@_register_op(FlattenOpr)
@_fallback_name
def flatten(opr: FlattenOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.flatten,
        args=(load_arg(opr.inp_tensors[0]),),
    )
    return [node]


@_register_op(RepeatOpr)
@_fallback_name
def repeat(opr: RepeatOpr, graph: torch.fx.Graph, load_arg: Callable, name: str):
    node = make_func_node(
        graph=graph,
        name=name,
        func=torch.repeat_interleave,
        args=(load_arg(opr.inp_tensors[0]),),
        repeats=opr.repeats,
        dim=opr.axis,
    )
    return [node]


@_register_op(PixelShuffle)
@_fallback_name
def pixshuffle(opr: PixelShuffle, graph: torch.fx.Graph, load_arg: Callable, name: str):
    # for mtk
    block_size = opr.scale_factor
    n, c, h, w = opr.inp_tensors[0].shape
    if opr.mode == "downsample":
        node = make_method_node(
            graph=graph,
            name=name + "_view",
            func="view",
            args=(
                load_arg(opr.inp_tensors[0]),
                (n, c, h // block_size, block_size, w // block_size, block_size),
            ),
        )
        node = make_method_node(
            graph=graph,
            name=name + "_permute",
            func="permute",
            args=(node, [0, 3, 5, 1, 2, 4]),
        )
        node = make_method_node(
            graph=graph, name=name + "_contiguous", func="contiguous", args=(node,),
        )
        node = make_method_node(
            graph=graph,
            name=name,
            func="view",
            args=(node, (n, c * (block_size ** 2), h // block_size, w // block_size)),
        )
    else:
        node = make_method_node(
            graph=graph,
            name=name + "_view",
            func="view",
            args=(
                load_arg(opr.inp_tensors[0]),
                (n, block_size, block_size, c // (block_size ** 2), h, w),
            ),
        )
        node = make_method_node(
            graph=graph,
            name=name + "_permute",
            func="permute",
            args=(node, [0, 3, 4, 1, 5, 2]),
        )
        node = make_method_node(
            graph=graph, name=name + "_contiguous", func="contiguous", args=(node,),
        )
        node = make_method_node(
            graph=graph,
            name=name,
            func="view",
            args=(node, (n, c // (block_size ** 2), h * block_size, w * block_size)),
        )
    return [node]
