import numpy as np
import tensorrt as trt
import math

from mgeconvert.frontend.mge_to_ir.op_generators import tensor
from .utils import add_missing_trt_tensors,mge_dtype_to_trt,broadcast_trt_tensors, mge_dim_to_trt_axes
from ...frontend.mge_to_ir.mge_utils import get_symvar_value
from ...converter_ir.ir_op import (
    AbsOpr,
    AdaptiveAvgPool2dOpr,
    AddOpr,
    AvgPool2dOpr,
    BroadcastOpr,
    BatchNormalizationOpr,
    CeilOpr,
    ConcatOpr,
    LeakyReluOpr,
    ConstantOpr,
    Conv2dOpr,
    Deconv2dOpr,
    ExpOpr,
    FlattenOpr,
    FloorOpr,
    FuseMulAdd3Opr,
    GetSubTensorOpr,
    GetVarShapeOpr,
    # GeluOpr,
    HardSigmoidOpr,
    HardSwishOpr,
    IdentityOpr,
    LinearOpr,
    LinspaceOpr,
    LogOpr,
    MatMulOpr, 
    MaxOpr,
    MaxPool2dOpr,
    MinOpr,
    MulOpr,
    MultipleDeviceTensorHolderOpr,
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
    TransposeOpr, #
    TrueDivOpr,
    TypeCvtOpr,
)

MGE2TRT = {}

def _register_op(*oprs):
    def callback(impl):
        for opr in oprs:
            MGE2TRT[opr] = impl
        return impl

    return callback

@_register_op(Conv2dOpr)
def _conv2d(mge_opr, network, var2tensor):
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]

    weight = inp_tensors[1]
    weight_shape = weight.shape

    kernel_size = (weight_shape[2], weight_shape[3])
    out_channels = weight_shape[0] 

    if len(weight_shape) == 5:
        kernel_size = (weight_shape[3], weight_shape[4])
        out_channels = weight_shape[0]*weight_shape[1]

    stride = mge_opr.stride  
    padding = mge_opr.padding
    dilation = mge_opr.dilation

    kernel = weight.np_data   
    
    bias = trt.Weights(mge_dtype_to_trt(weight.dtype)) 
    if len(inp_tensors) > 2:
        bias = inp_tensors[2].np_data
    layer = network.add_convolution(
            input=input_trt,
            num_output_maps=out_channels,
            kernel_shape=kernel_size,
            kernel=kernel,
            bias=bias)
    layer.stride = stride
    layer.padding = padding
    layer.dilation = dilation

    if mge_opr.groups is not None:
        layer.num_groups = mge_opr.groups

    return layer.get_output(0)

@_register_op(
    AddOpr,
    SubOpr,
    MulOpr,
    TrueDivOpr,
    MaxOpr,
    MinOpr,
    PowOpr
)
def _elsewise(mge_opr, network, var2tensor):
    support_op_map = {
        AddOpr: trt.ElementWiseOperation.SUM,
        SubOpr: trt.ElementWiseOperation.SUB,
        TrueDivOpr: trt.ElementWiseOperation.DIV,
        MulOpr: trt.ElementWiseOperation.PROD,
        PowOpr: trt.ElementWiseOperation.POW,
        MaxOpr: trt.ElementWiseOperation.MAX,
        MinOpr: trt.ElementWiseOperation.MIN,
    }
    inp_tensors = mge_opr.inp_tensors
    out_tensors = mge_opr.out_tensors
    output = out_tensors[0]
    input_a = inp_tensors[0]
    input_b = inp_tensors[1]
    input_a_trt, input_b_trt = add_missing_trt_tensors(network, [input_a, input_b], var2tensor)
    input_a_trt, input_b_trt = broadcast_trt_tensors(network, [input_a_trt, input_b_trt], len(output.shape)-1)
    layer = network.add_elementwise(input_a_trt, input_b_trt, support_op_map[type(mge_opr)])
    return layer.get_output(0)

@_register_op(
    AddOpr,
)
def _elsewise(mge_opr, network, var2tensor):
    support_op_map = {
        AddOpr: trt.ElementWiseOperation.SUM,
        SubOpr: trt.ElementWiseOperation.SUB,
        TrueDivOpr: trt.ElementWiseOperation.DIV,
        MulOpr: trt.ElementWiseOperation.PROD,
        PowOpr: trt.ElementWiseOperation.POW,
        MaxOpr: trt.ElementWiseOperation.MAX,
        MinOpr: trt.ElementWiseOperation.MIN,
    }
    inp_tensors = mge_opr.inp_tensors
    out_tensors = mge_opr.out_tensors
    output = out_tensors[0]
    input_a = inp_tensors[0]
    input_b = inp_tensors[1]
    input_a_trt, input_b_trt = add_missing_trt_tensors(network, [input_a, input_b], var2tensor)
    input_a_trt, input_b_trt = broadcast_trt_tensors(network, [input_a_trt, input_b_trt], len(output.shape)-1)
    layer = network.add_elementwise(input_a_trt, input_b_trt, support_op_map[type(mge_opr)])
    return layer.get_output(0)


@_register_op(
    ReluOpr,
    SigmoidOpr,
    TanHOpr,
)
def _active(mge_opr, network, var2tensor):
    support_op_map = {
        ReluOpr: trt.ActivationType.RELU,
        SigmoidOpr: trt.ActivationType.SIGMOID,
        TanHOpr:trt.ActivationType.TANH,
    }
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]

    layer = network.add_activation(
        input=input_trt, type=support_op_map[type(mge_opr)])
    return layer.get_output(0)


@_register_op(
    Relu6Opr
)
def _relu6(mge_opr, network, var2tensor):
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    out_tensors = mge_opr.out_tensors
    output = out_tensors[0]
    input_a_trt, input_b_trt = add_missing_trt_tensors(network, [input, 6], var2tensor)
    input_a_trt, input_b_trt = broadcast_trt_tensors(network, [input_a_trt, input_b_trt], len(output.shape)-1)
    layer = network.add_activation(
        input=input_a_trt, type=trt.ActivationType.RELU)
    layer = network.add_elementwise(
        layer.get_output(0), input_b_trt, trt.ElementWiseOperation.MIN)
    return layer.get_output(0)


# @_register_op(
#     GeluOpr
# )
# def _gelu(mge_opr, network, var2tensor):

#     inp_tensors = mge_opr.inp_tensors
#     input = inp_tensors[0]
#     out_tensors = mge_opr.out_tensors
#     output = out_tensors[0]
    
#     x, c05, c1, cs2pi, c044, c3 = add_missing_trt_tensors(
#         network,
#         [input, 0.5, 1.0, math.sqrt(2.0 / math.pi), 0.044715, 3.0],
#         var2tensor
#     )
    
#     x, c05, c1, cs2pi, c044, c3 = broadcast_trt_tensors(
#         network, 
#         [x, c05, c1, cs2pi, c044, c3], 
#         len(output.shape) - 1
#     )
    
#     y = network.add_elementwise(x, c3, trt.ElementWiseOperation.POW).get_output(0)
#     y = network.add_elementwise(y, c044, trt.ElementWiseOperation.PROD).get_output(0)
#     y = network.add_elementwise(x, y, trt.ElementWiseOperation.SUM).get_output(0)
#     y = network.add_elementwise(y, cs2pi, trt.ElementWiseOperation.PROD).get_output(0)
#     y = network.add_activation(y, trt.ActivationType.TANH).get_output(0)
#     y = network.add_elementwise(y, c1, trt.ElementWiseOperation.SUM).get_output(0)
#     y = network.add_elementwise(x, y, trt.ElementWiseOperation.PROD).get_output(0)
#     y = network.add_elementwise(y, c05, trt.ElementWiseOperation.PROD).get_output(0)
    
#     output._trt = y


@_register_op(
    SoftmaxOpr
)
def _softmax(mge_opr, network, var2tensor):
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]
    import pdb
    pdb.set_trace()
    dim = mge_opr.axis
    if dim < 0:
        dim = len(input.shape) + dim
    axes = 1 << (dim - 1)
    layer = network.add_softmax(input=input_trt)
    layer.axes = axes
    return layer.get_output(0)


@_register_op(
    LeakyReluOpr,
)
def _leakyrelu(mge_opr, network, var2tensor):
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]

    negative_slope = mge_opr.negative_slope
    
    layer = network.add_activation(input_trt, trt.ActivationType.LEAKY_RELU)
    layer.alpha = negative_slope
    return layer.get_output(0)


@_register_op(
    AbsOpr,
    ExpOpr,
    LogOpr,
    FloorOpr,
    CeilOpr,
)
def _unary(mge_opr, network, var2tensor):
    support_op_map = {
        AbsOpr: trt.UnaryOperation.ABS,
        ExpOpr: trt.UnaryOperation.EXP,
        LogOpr: trt.UnaryOperation.LOG,
        FloorOpr:trt.UnaryOperation.FLOOR,
        CeilOpr:trt.UnaryOperation.CEIL,
    }
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]
    
    layer = network.add_unary(input_trt, support_op_map[type(mge_opr)])
    return layer.get_output(0)


@_register_op(
    ReshapeOpr,
)
def _reshape(mge_opr, network, var2tensor):
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]
    # input_trt = var2tensor[input]

    out_shape = mge_opr.out_tensors[0].shape
    layer = network.add_shuffle(input_trt)
    layer.reshape_dims = tuple(out_shape[1:])
    return layer.get_output(0)


@_register_op(
    TransposeOpr,
)
def convert_transpose(mge_opr, network, var2tensor):
    pattern = mge_opr.pattern
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]

    assert(pattern[0] == 0)  # cannot move batch dim

    trt_pattern = tuple([p - 1 for p in pattern])[1:]
    layer = network.add_shuffle(input_trt)
    layer.second_transpose = tuple(trt_pattern)

    return layer.get_output(0)


@_register_op(
    FlattenOpr,
)
def _convert_transpose(mge_opr, network, var2tensor):
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]
    out_tensors = mge_opr.out_tensors
    output = out_tensors[0]

    layer = network.add_shuffle(input_trt)
    layer.reshape_dims = tuple(output.shape[1:])

    return layer.get_output(0)


@_register_op(
    ReduceOpr,
)
def _reduce(mge_opr, network, var2tensor):
    support_op_map = {
        "SUM": trt.ReduceOperation.SUM,
        "MEAN": trt.ReduceOperation.AVG,
        "MAX": trt.ReduceOperation.MAX,
        "MIN": trt.ReduceOperation.MIN,
    }
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]
    dim = mge_opr.axis
    keepdim = mge_opr.keep_dims

    layer = network.add_reduce(input_trt, support_op_map[mge_opr.mode], mge_dim_to_trt_axes(dim), keepdim)
    return layer.get_output(0)


@_register_op(
    MaxPool2dOpr,
    AvgPool2dOpr,
)
def _pool2d(mge_opr, network, var2tensor):
    support_op_map = {
        MaxPool2dOpr: trt.PoolingType.MAX,
        AvgPool2dOpr: trt.PoolingType.AVERAGE,
    }
    
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    kernel_size = mge_opr.kernel_size
    stride = mge_opr.stride
    padding = mge_opr.padding
    out_tensors = mge_opr.out_tensors
    output = out_tensors[0]
    
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]

    kernel_size = tuple(kernel_size)

    stride = tuple(stride)

    padding = tuple(padding)

    layer = network.add_pooling(
        input=input_trt, type=support_op_map[type(mge_opr)], window_size=kernel_size)
    
    layer.stride = stride
    layer.padding = padding

    if isinstance(mge_opr,AvgPool2dOpr):
        mode = mge_opr.mode
        layer.average_count_excludes_padding = True if mode == "AVERAGE_COUNT_EXCLUDE_PADDING" else False

    return layer.get_output(0)


@_register_op(
    LinearOpr,
    MatMulOpr,
)
def _full_connect(mge_opr, network, var2tensor):
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    weight = inp_tensors[1]
    bias = trt.Weights(mge_dtype_to_trt(weight.dtype)) 
    if len(inp_tensors) > 2:
        bias = inp_tensors[2].np_data

    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]
    out_tensors = mge_opr.out_tensors
    output = out_tensors[0]

    layer = network.add_shuffle(input_trt)
    layer.reshape_dims = tuple(input_trt.shape) + (1, 1) 
        
    layer = network.add_fully_connected(
        input=layer.get_output(0),
        num_outputs=int(weight.shape[0]),
        kernel=weight.np_data,
        bias=bias)

    layer = network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = tuple(output.shape[1:])

    return layer.get_output(0)

@_register_op(
    TypeCvtOpr,
)
def _type_cvt(mge_opr, network, var2tensor):

    inp_tensors = mge_opr.inp_tensors
    out_dtype = mge_opr.out_dtype
    input = inp_tensors[0]
    shape = tuple(input.shape[1:])
    weight = input.np_data
    layer = network.add_constant(shape, weight)
    layer.type = mge_dtype_to_trt(out_dtype)
    import pdb
    pdb.set_trace()
    return layer.get_output(0)


@_register_op(
    IdentityOpr,
)
def _identity(mge_opr, network, var2tensor):
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]
    out_tensors = mge_opr.out_tensors
    output = out_tensors[0]
    return input_trt


@_register_op(
    ConcatOpr,
)
def _convert_cat(mge_opr, network, var2tensor):
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    dim = mge_opr.axis
    out_tensors = mge_opr.out_tensors
    output = out_tensors[0]

    if dim < 0:
        dim = len(input.shape) - abs(dim)
    
    assert dim != 0,"cannot move batch dim"
    input_trts = add_missing_trt_tensors(network, inp_tensors, var2tensor)
    trt_inputs = broadcast_trt_tensors(network, input_trts, len(output.shape) - 1)

    layer = network.add_concatenation(inputs=trt_inputs)
    layer.axis = dim - 1

    return layer.get_output(0)

@_register_op(
    GetSubTensorOpr,
)
def _convert_subtensor(mge_opr, network, var2tensor):
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    input_trt = var2tensor[input]
    len_input = len(input.shape)
    axis = mge_opr.axis  ##[]
    begin_params = mge_opr.begin_params
    step_params = mge_opr.step_params
    squeeze_axis = mge_opr.squeeze_axis
    out_tensors = mge_opr.out_tensors
    output_shape = list(out_tensors[0].shape)
    full_output_shape = output_shape.copy()
    if squeeze_axis:
        for i in squeeze_axis:
            full_output_shape.insert(i,1)
    assert len(full_output_shape) == len(input.shape)
    sizes = full_output_shape[1:]
    starts = [0]*(len_input-1)
    strides = [1]*(len_input-1)
    for i in range(len(axis)-1,-1,-1):
        if axis[i] == 0:
            continue
        starts[axis[i]-1] = begin_params[i]
        strides[axis[i]-1] = step_params[i]

    output_trt = network.add_slice(input_trt, starts, sizes, strides).get_output(0)


    if squeeze_axis:
        layer = network.add_shuffle(output_trt)
        layer.reshape_dims = tuple(output_shape[1:]) # exclude batch
        output_trt = layer.get_output(0)

    return output_trt


@_register_op(
    AdaptiveAvgPool2dOpr,
)
def _convert_AdaptiveAvgPool2d(mge_opr, network, var2tensor):

    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    out_tensors = mge_opr.out_tensors
    output = out_tensors[0]
    shape = inp_tensors[1].np_data
    
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]

    output_size = (shape[0], shape[1])

    stride = (input_trt.shape[-2] // output_size[-2], input_trt.shape[-1] // output_size[-1])

    kernel_size = stride
    layer = network.add_pooling(
        input=input_trt, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    layer.stride = stride

    return layer.get_output(0)


@_register_op(
    BatchNormalizationOpr,
)
def _BatchNorm2d(mge_opr, network, var2tensor):

    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]
    weight = inp_tensors[1]
    bias = inp_tensors[2]
    running_mean = inp_tensors[3]
    running_var = inp_tensors[4]
    eps = mge_opr.eps

    scale = weight.np_data / np.sqrt(
        running_var.np_data + eps
    )
    bias = (
        bias.np_data
        - running_mean.np_data * scale
    )

    power = np.ones_like(scale)

    layer = network.add_scale(input_trt, trt.ScaleMode.CHANNEL, bias, scale, power)

    return layer.get_output(0)



@_register_op(
    HardSwishOpr,
)
def _convert_HardSwish(mge_opr, network, var2tensor):

    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    shape = (1,) * (len(input.shape) - 1)
    input_trt = add_missing_trt_tensors(network, [input], var2tensor)[0]

    # x + 3
    tensor = 3.0 * np.ones(shape, dtype=np.float32)
    trt_3  = network.add_constant(shape, tensor)
    tmp    = network.add_elementwise(input_trt, trt_3.get_output(0), trt.ElementWiseOperation.SUM)

    # relu6(x + 3)
    relu   = network.add_activation(input=tmp.get_output(0), type=trt.ActivationType.RELU)
    tensor = 6.0 * np.ones(shape, dtype=np.float32)
    trt_6  = network.add_constant(shape, tensor)
    relu_6 = network.add_elementwise(relu.get_output(0), trt_6.get_output(0), trt.ElementWiseOperation.MIN)

    # x * relu6(x + 3)
    tmp    = network.add_elementwise(input_trt, relu_6.get_output(0), trt.ElementWiseOperation.PROD)

    # x * relu6(x + 3) / 6
    layer = network.add_elementwise(tmp.get_output(0), trt_6.get_output(0), trt.ElementWiseOperation.DIV)

    return layer.get_output(0)



@_register_op(
    GetVarShapeOpr,
)
def _GetVarShape(mge_opr, network, var2tensor):
    inp_tensors = mge_opr.inp_tensors
    input = inp_tensors[0]
    out_tensors = mge_opr.out_tensors
    output = out_tensors[0]
    shape = output.shape
    if hasattr(output, "_var"):
        tensor = get_symvar_value(output._var)
    else:
        tensor = np.array(input.shape, dtype=np.int64)
    
    layer = network.add_constant(shape, tensor)

    return layer.get_output(0)