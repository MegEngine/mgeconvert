from megengine.functional import ones
from megengine.core._imperative_rt.core2 import Tensor as RawTensor

import tensorrt as trt
import numpy as np
from typing import NamedTuple, Iterable, Tuple

from mgeconvert.converter_ir.ir_tensor import IRTensor


def mge_dtype_to_trt(dtype):
    if trt.__version__ >= '7.0' and dtype == bool:
        return trt.bool
    elif dtype == np.int8:
        return trt.int8
    elif dtype == np.int32:
        return trt.int32
    elif dtype == np.float16:
        return trt.float16
    elif dtype == np.float32:
        return trt.float32
    else:
        raise TypeError("%s is not supported by tensorrt" % dtype)

def mge_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return np.int8
    elif trt.__version__ >= '7.0' and dtype == trt.bool:
        return bool
    elif dtype == trt.int32:
        return np.int32
    elif dtype == trt.float16:
        return np.float16
    elif dtype == trt.float32:
        return np.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)

def mge_device_to_trt(device_type):
    if device_type == "gpu":
        return trt.TensorLocation.DEVICE
    elif device_type == "cpu":
        return trt.TensorLocation.HOST
    else:
        return TypeError("%s is not supported by tensorrt" % device_type)

def mge_dim_to_trt_axes(dim):
    if not isinstance(dim, tuple):
        dim = (dim,)

    axes = 0
    for d in dim:
        axes |= 1<< (d-1)

    return axes

class InputTensorSpec(NamedTuple):
    shape : Tuple
    dtype : np.dtype
    has_batch_dim : bool = True

    @classmethod
    def from_tensor(cls, tensor: RawTensor):
        return cls(tensor.shape, tensor.dtype)

    @classmethod
    def from_tensors(cls, tensors: Iterable[RawTensor]):
        return [cls.from_tensor(t) for t in tensors]

def check_dtype(*tensors):
    dtype = None
    for t in tensors:
        if isinstance(t, IRTensor):
            if dtype is None:
                dtype = t.dtype
            else:
                assert dtype == t.dtype  # , 'TensorNode dtypes must match')
    assert (
        dtype is not None
    )  # , 'Data type could not be inferred from any item in list')
    return dtype

def add_missing_trt_tensors(network, tensors, var2tensor):
    """Creates missing TensorRT tensors as constants and attaches them to the TensorNode"""
    trt_tensors = [None] * len(tensors)

    dtype = check_dtype(*tensors)

    for i, t in enumerate(tensors):
        trt_tensor = None

        # if t.np_data is not None:
        #     shape = t.np_data.shape
        #     scalar = t.np_data * ones(shape, dtype=dtype).numpy()
        #     trt_tensor = network.add_constant(shape, scalar).get_output(0)
        if t in var2tensor:
            trt_tensor = var2tensor[t]
        else:
            # remove all preceding ones, these can be re-inserted later when broadcasting
            num_preceding_ones = 0
            for j in range(len(t.shape)):
                if int(t.shape[j]) == 1:
                    num_preceding_ones += 1
                else:
                    break
            shape = tuple(t.shape[num_preceding_ones:])
            weight = t.np_data
            var2tensor[t] = network.add_constant(shape, weight).get_output(0)
            trt_tensor = var2tensor[t]

        assert trt_tensor is not None

        trt_tensors[i] = trt_tensor

    return trt_tensors

def broadcast_trt_tensors(network, trt_tensors, broadcast_ndim):
    """Broadcast TensorRT tensors to the specified dimension by pre-padding shape 1 dims"""
    broadcasted_trt_tensors = [None] * len(trt_tensors)
    
    for i, t in enumerate(trt_tensors):
        if len(t.shape) < broadcast_ndim:
            # append 1 size dims to front
            diff = broadcast_ndim - len(t.shape)
            shape = tuple([1] * diff + list(t.shape))
            layer = network.add_shuffle(t)
            layer.reshape_dims = shape
            trt_tensor = layer.get_output(0)
        else:
            trt_tensor = t

        broadcasted_trt_tensors[i] = trt_tensor
        
    return broadcasted_trt_tensors

def default_irtensor_names(irtensor_list):
    return ["irtensor_"+irtensor.name for irtensor in irtensor_list]

def get_dynamic_dims(shape):
    dynamic_dims = []

    for i, s in enumerate(shape):
        if s == -1:
            dynamic_dims.append(i)

    return dynamic_dims