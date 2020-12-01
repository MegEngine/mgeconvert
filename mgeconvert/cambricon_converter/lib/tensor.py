# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import logging
from enum import Enum, unique
from functools import reduce

import numpy as np

from .cnlib import cambriconLib as cnlib


@unique
class HEAD(Enum):
    UNINITIALIZED = 1
    CPU = 2
    MLU = 3


@unique
class TENSOR_TYPE(Enum):
    TENSOR = 0
    FILTER = 1
    CONST = 2
    VARIABLE = 3


@unique
class TENSOR_LAYOUT(Enum):
    NCHW = 0
    NHWC = 3


@unique
class DATA_TYPE(Enum):
    FLOAT16 = 1
    FLOAT32 = 2
    INT8 = 4
    UINT8 = 8


class Tensor:
    _map_tensor_type = {
        TENSOR_TYPE.TENSOR: cnlib.CNML_TENSOR,  # 0
        TENSOR_TYPE.FILTER: cnlib.CNML_FILTER,  # 1
        TENSOR_TYPE.CONST: cnlib.CNML_CONST,  # 2
        TENSOR_TYPE.VARIABLE: cnlib.CNML_VARIABLE,  # 3
    }

    _map_cn_tensor_layout = {
        TENSOR_LAYOUT.NCHW: cnlib.CNML_NCHW,  # 0
        TENSOR_LAYOUT.NHWC: cnlib.CNML_NHWC,  # 3
    }

    _map_sizeof = {
        DATA_TYPE.INT8: 1,
        DATA_TYPE.UINT8: 1,
        DATA_TYPE.FLOAT16: 2,
        DATA_TYPE.FLOAT32: 4,
    }

    _map_np_data_type = {
        DATA_TYPE.INT8: np.int8,
        DATA_TYPE.UINT8: np.uint8,
        # fp32 -> fp16 at cpp level
        DATA_TYPE.FLOAT16: np.float32,
        DATA_TYPE.FLOAT32: np.float32,
    }

    _map_cn_data_type = {
        DATA_TYPE.FLOAT16: cnlib.CNML_DATA_FLOAT16,  # 1
        DATA_TYPE.FLOAT32: cnlib.CNML_DATA_FLOAT32,  # 2
        DATA_TYPE.INT8: cnlib.CNML_DATA_INT8,  # 4
        DATA_TYPE.UINT8: cnlib.CNML_DATA_UINT8,  # 8
    }

    _head = None

    _cnml_tensor = None

    _tensor_type = None
    _cn_tensor_type = None

    _tensor_layout = None
    _cn_tensor_layout = None

    _data_type = None
    _cn_data_type = None

    _data_count = 0
    _data_bytes = 0

    _data_mlu = None
    _data_cpu = None

    _scale = None
    _cn_position = None
    _cn_scale = None

    _name = None

    _default_dtype = DATA_TYPE.FLOAT32

    NCHW2NHWC = False

    MUTABLE = False

    # best practice: always initilize data at the beginning
    def __init__(
        self,
        shape,
        ttype=TENSOR_TYPE.TENSOR,
        tlayout=TENSOR_LAYOUT.NCHW,
        dtype=None,
        name=None,
        data=None,
        scale=None,
        quantized=False,
    ):
        self._name = name
        self.quantized = quantized

        # create cnml_tensor by tensor type
        self._tensor_type = ttype
        self._cn_tensor_type = self._map_tensor_type[ttype]
        if self._cn_tensor_type is not cnlib.CNML_VARIABLE:
            self._cnml_tensor = cnlib.cnTensor_V2(self._cn_tensor_type)
            if Tensor.MUTABLE and ttype == TENSOR_TYPE.TENSOR:
                cnlib.cnmlSetTensorDimMutable(
                    self._cnml_tensor, [True, False, False, False]
                )
        else:
            self._cnml_tensor = cnlib.cnTensor_V3()
            cnlib.cnmlSetTensorType(self._cnml_tensor, cnlib.CNML_VARIABLE)

        # set cnml_tensor data type
        if dtype is None:
            dtype = Tensor._default_dtype
        if dtype not in self._map_cn_data_type.keys():
            raise NotImplementedError(
                "host data type {} was not supported now.".format(self.dtype)
            )
        self._data_type = dtype
        self._cn_data_type = self._map_cn_data_type[dtype]
        cnlib.cnmlSetTensorDataType(self._cnml_tensor, self._cn_data_type)

        if Tensor.NCHW2NHWC:
            tlayout = TENSOR_LAYOUT.NHWC
        # set layout
        if tlayout not in self._map_cn_tensor_layout.keys():
            raise NotImplementedError(
                "host tensor layout {} was not supported now.".format(tlayout)
            )
        self._tensor_layout = tlayout
        self._cn_tensor_layout = self._map_cn_tensor_layout[tlayout]

        # set cnml_tensor shape
        shape = list(shape)
        if len(shape) < 4:
            shape += [1] * (4 - len(shape))
        self._data_shape = shape
        if Tensor.NCHW2NHWC:
            cnlib.cnmlSetTensorShape(
                self._cnml_tensor, self._get_internal_shape(self._data_shape)
            )
        else:
            cnlib.cnmlSetTensorShape(self._cnml_tensor, self._data_shape)

        # set cnml_tensor position and scale
        if scale is not None:
            self._set_scale(scale)

        if self._cn_data_type == cnlib.CNML_DATA_INT8:
            assert scale is not None, "quantized tensor must set scale"
            cnlib.cnmlSetQuantizedPosition(self._cnml_tensor, self._cn_position)
            cnlib.cnmlSetQuantizedScale(self._cnml_tensor, self._cn_scale)

        # _data_mlu: cpu->mlu for inputs, mlu->cpu for outputs
        self._data_count = reduce(lambda x, y: x * y, self._data_shape, 1)
        self._data_bytes = self._data_count * self._map_sizeof[self._data_type]
        if self._cn_tensor_type is cnlib.CNML_TENSOR:
            self._data_mlu = cnlib.cnMalloc(self._data_bytes)

        if self._cn_tensor_type in (cnlib.CNML_FILTER, cnlib.CNML_CONST):
            assert data is not None, "data must be specified for filter/const tensor"

        if data is None:
            self._head = HEAD.UNINITIALIZED
        else:
            self._set_data_cpu(data)

    def _get_quantized_info(self, scale):
        assert isinstance(scale, (float, int))
        if scale < 2 ** -32 or scale > 2 ** 32:
            logging.warning(
                "scale is not in the valid range:(%s, %s), which will result in "
                "incorrect calculation of the converted model.",
                2 ** -32,
                2 ** 32,
            )
        summ = 1.0
        cn_position = 0
        if scale >= 1.0:
            while summ < scale:
                cn_position = cn_position + 1
                summ = summ * 2
        elif scale > 0.0:
            while summ >= scale:
                cn_position = cn_position - 1
                summ = summ * 0.5
            cn_position = cn_position + 1
            summ = summ * 2
        else:
            raise ValueError("scale {} should > 0".format(scale))

        if cn_position > 32:
            cn_position = 32
        if cn_position < -32:
            cn_position = -32

        cn_scale = float(2 ** cn_position) / scale

        return cn_position, cn_scale

    # NCHW -> NHWC
    def _get_internal_shape(self, shape):
        internal_shape = list(shape)
        internal_shape[1], internal_shape[2], internal_shape[3] = (
            internal_shape[2],
            internal_shape[3],
            internal_shape[1],
        )
        return internal_shape

    @property
    def mludata(self):
        return self._data_mlu

    @property
    def cpudata(self):
        if self._data_cpu is None:
            return None
        if Tensor.NCHW2NHWC:
            ret = self._data_cpu.astype(self._map_np_data_type[self._data_type])
            ret = ret.reshape(self._get_internal_shape(self._data_shape))
            ret = ret.transpose((0, 3, 1, 2))
        else:
            ret = self._data_cpu.reshape(self._data_shape)
        return ret

    @cpudata.setter
    def cpudata(self, data):
        if data is None:
            self._data_cpu = None
            return

        if self._cn_tensor_type in (cnlib.CNML_CONST, cnlib.CNML_FILTER):
            return

        self._set_data_cpu(data)

    def _set_data_cpu(self, data):
        if isinstance(data, (int, float)):
            data = [data] * self._data_count
        data = (
            np.asarray(data)
            .astype(self._map_np_data_type[self._data_type])
            .reshape(self._data_shape)
        )
        if Tensor.NCHW2NHWC:
            data = data.transpose((0, 2, 3, 1))
        data = data.flatten()

        self._data_cpu = data
        self._head = HEAD.CPU

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        if scale is None:
            self._scale = None
            self._cn_position = None
            self._cn_scale = None
        self._set_scale(scale)

    def _set_scale(self, scale):
        self._scale = scale
        self._cn_position, self._cn_scale = self._get_quantized_info(self._scale)

    @property
    def head(self):
        return self._head

    @property
    def shape(self):
        return self._data_shape

    @property
    def name(self):
        return self._name

    @property
    def ttype(self):
        return self._tensor_type

    @property
    def tlayout(self):
        return self._tensor_layout

    @property
    def dtype(self):
        return self._data_type

    @property
    def cnmlTensor(self):
        return self._cnml_tensor

    def h2d(self):
        if self._data_cpu is not None:
            if self._cn_tensor_type in (cnlib.CNML_FILTER, cnlib.CNML_CONST):
                if self._cn_data_type is cnlib.CNML_DATA_INT8:
                    cnlib.cnH2dConstInt8(self._cnml_tensor, self._data_cpu, False)
                elif self._cn_data_type is cnlib.CNML_DATA_FLOAT16:
                    cnlib.cnH2dConstFloat16(self._cnml_tensor, self._data_cpu, False)
                else:
                    cnlib.cnH2dConst(self._cnml_tensor, self._data_cpu, False)
            else:
                # cpu->mlu: for inputs
                if self._cn_data_type is cnlib.CNML_DATA_UINT8:
                    cnlib.cnH2dUint8(self._data_mlu, self._data_cpu)
                elif self._cn_data_type is cnlib.CNML_DATA_FLOAT16:
                    cnlib.cnH2dFloat16(self._data_mlu, self._data_cpu)
                else:
                    cnlib.cnH2d(self._data_mlu, self._data_cpu)
            self._head = HEAD.MLU

    def d2h(self):
        if self._data_mlu is not None:
            self._head = HEAD.MLU
            if self._data_cpu is None:
                self._data_cpu = np.array([0.0] * self._data_count).astype(
                    self._map_np_data_type[self._data_type]
                )
            # mlu->cpu: for outputs
            if self._cn_data_type is cnlib.CNML_DATA_FLOAT16:
                cnlib.cnD2hFloat16(self._data_cpu, self._data_mlu)
            else:
                cnlib.cnD2h(self._data_cpu, self._data_mlu)

    def ensure_mlu(self):
        if self._head != HEAD.MLU:
            self.h2d()

    def destroy(self):
        if self._data_mlu is not None:
            cnlib.cnrtFree(self._data_mlu)
        cnlib.cnDestroyTensor(self._cnml_tensor)
