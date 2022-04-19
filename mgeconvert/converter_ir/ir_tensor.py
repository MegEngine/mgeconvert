# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import List, Sequence


class DataFormat:
    @staticmethod
    def permute_shape(shape, order):
        return [shape[i] for i in order]


class NHWCFormat:
    def __init__(self) -> None:
        self.permute_to_NCHW = [0, 3, 1, 2]

    def shape_to_NCHW(self, data):
        assert isinstance(data.axis_order, NHWCFormat)
        assert data.ndim == 4
        return DataFormat.permute_shape(data.shape, self.permute_to_NCHW)

    def data_to_NCHW(self, data):
        assert data.ndim == 4
        return data.transpose(self.permute_to_NCHW)


class NCHWFormat:
    def __init__(self):
        self.permute_to_NHWC = [0, 2, 3, 1]

    def shape_to_NHWC(self, shape):
        assert len(shape) == 4
        return DataFormat.permute_shape(shape, self.permute_to_NHWC)

    def data_to_NHWC(self, data):
        assert data.ndim == 4
        return data.transpose(self.permute_to_NHWC)


class IOHWFormat:
    def __init__(self):
        self.permute_to_OHWI = [1, 2, 3, 0]

    def shape_to_OHWI(self, shape):
        assert len(shape) == 4
        return DataFormat.permute_shape(shape, self.permute_to_OHWI)

    def data_to_OHWI(self, data):
        assert data.ndim == 4
        return data.transpose(self.permute_to_OHWI)


class OIHWFormat(NCHWFormat):
    def shape_to_OHWI(self, shape):
        return super().shape_to_NHWC(shape)

    def shape_to_IHWO(self, shape):
        return [shape[1], shape[2], shape[3], shape[0]]

    def data_to_OHWI(self, data):
        return super().data_to_NHWC(data)

    def data_to_IHWO(self, data):
        return data.transpose(1, 2, 3, 0)


class OHWIFormat(NHWCFormat):
    def shape_to_OIHW(self, data):
        return super().shape_to_NCHW(data)

    def data_to_OIHW(self, data):
        return super().data_to_NCHW(data)


class AxisOrder:
    NCHW = NCHWFormat()
    NHWC = NHWCFormat()

    OIHW = OIHWFormat()
    OHWI = OHWIFormat()

    IOHW = IOHWFormat()  # deconv weight


class IRTensor:
    def __init__(
        self,
        name,
        shape,
        dtype,
        scale=None,
        zero_point=None,
        qmin=None,
        qmax=None,
        q_type=None,
        np_dtype=None,
        np_data=None,
        owner_opr=None,
        axis=AxisOrder.NCHW,
    ):
        self.name = name
        self.axis_order = axis
        self.owner_opr = owner_opr
        self.user_opr = []
        self.shape = shape
        self.dtype = dtype

        self.np_data = np_data

        self.scale = scale
        self.zero_point = zero_point
        self.qmin = qmin
        self.qmax = qmax
        assert isinstance(q_type, str) or q_type is None
        self.q_dtype = q_type
        self.np_dtype = np_dtype

    @property
    def ndim(self):
        return len(self.shape)

    def add_user_opr(self, op):
        self.user_opr.append(op)

    def set_dtype(self, target_type):
        self.np_data = self.np_data.astype(target_type)
        self.dtype = target_type

    def set_qparams_from_other_tensor(self, other):
        self.q_dtype = other.q_dtype
        self.np_dtype = other.np_dtype
        self.qmin = other.qmin
        self.qmax = other.qmax
        self.scale = other.scale
        self.zero_point = other.zero_point

    def set_qparams_from_mge_qparams(self, qparams):
        dtype_meta = qparams.dtype_meta
        self.q_dtype = dtype_meta.name
        self.np_dtype = dtype_meta.np_dtype_str
        self.qmin = dtype_meta.qmin
        self.qmax = dtype_meta.qmax
        scale = qparams.scale
        zero_point = qparams.zero_point
        if not isinstance(scale, Sequence):  # per tensor
            self.scale = float(scale)
        else:  # per channel
            self.scale = [float(s) for s in scale]

        if zero_point is not None:
            if not isinstance(zero_point, Sequence):
                self.zero_point = int(zero_point)
            else:
                self.zero_point = [int(zp) for zp in zero_point]

    def set_qparams(
        self,
        *,
        scale: float,
        q_dtype: str,
        qmin: int = None,
        qmax: int = None,
        zero_point=None,
        np_dtype=None,
    ):
        if qmin is None or qmax is None:
            assert np_dtype is not None, "must provide np_dtype or qmin and qmax"
        if not isinstance(scale, Sequence):  # per tensor
            self.scale = float(scale)
        else:  # per channel
            self.scale = [float(s) for s in scale]
        if zero_point is not None:
            if not isinstance(zero_point, Sequence):
                self.zero_point = int(zero_point)
            else:
                self.zero_point = [int(zp) for zp in zero_point]

        self.q_dtype = q_dtype
        self.np_dtype = np_dtype
        self.qmin = qmin
        self.qmax = qmax
