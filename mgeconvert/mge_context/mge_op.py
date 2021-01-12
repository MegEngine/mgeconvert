# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import json
import sys

from .mge_utils import get_mge_version, get_opr_type, get_shape, get_symvar_value

mge_version = get_mge_version()


def str_to_mge_class(classname):
    return getattr(sys.modules[__name__], classname)


class MgeOpr:
    def __init__(self, opr):
        self._opr = opr
        self.name = opr.name

        self.name = self.name.replace(":", "_")
        self.name = self.name.replace(".", "_")
        self.name = self.name.replace(",", "_")

        self.id = opr.id
        self.type = get_opr_type(opr)
        self.flag = None
        self.inp_vars = []
        self.out_vars = []
        self.inp_oprs = []
        self.out_oprs = []
        self.params = opr.params if mge_version <= "0.6.0" else json.loads(opr.params)

    def add_inp_var(self, x):
        self.inp_vars.append(x)

    def add_out_var(self, x):
        if ":workspace" in x.name:
            return
        self.out_vars.append(x)

    def add_inp_opr(self, x):
        self.inp_oprs.append(x)

    def add_out_opr(self, x):
        self.out_oprs.append(x)

    @property
    def next_opr(self):
        assert len(self.out_oprs) == 1
        return self.out_oprs[0]

    @property
    def prev_opr(self):
        assert len(self.inp_oprs) == 1
        return self.inp_oprs[0]


class PoolingForwardOpr(MgeOpr):
    name = "PoolingForward"

    def __init__(self, opr):
        super().__init__(opr)
        self.data_format = self.params["format"]
        self.mode = self.params["mode"]

        self.ph, self.pw = self.params["pad_h"], self.params["pad_w"]
        self.sh, self.sw = self.params["stride_h"], self.params["stride_w"]
        self.kh, self.kw = self.params["window_h"], self.params["window_w"]


class MatrixMulOpr(MgeOpr):
    name = "MatrixMul"

    def __init__(self, opr):
        super().__init__(opr)
        self.format = self.params["format"]
        self.transposeB = self.params["transposeB"]
        self.transposeA = self.params["transposeA"]
        self.compute_mode = self.params["compute_mode"]


class BatchedMatrixMulOpr(MgeOpr):
    name = "BatchedMatrixMul"

    def __init__(self, opr):
        super().__init__(opr)
        self.format = self.params["format"]
        self.transposeA = self.params["transposeA"]
        self.transposeB = self.params["transposeB"]
        self.compute_mode = self.params["compute_mode"]


class ConcatOpr(MgeOpr):
    name = "Concat"

    def __init__(self, opr):
        super().__init__(opr)
        self.axis = self.params["axis"]


class ReshapeOpr(MgeOpr):
    name = "Reshape"

    def __init__(self, opr):
        super().__init__(opr)
        self.input_shape = get_shape(opr.inputs[0])
        self.output_shape = get_shape(opr.outputs[0])
        self.shape_param = get_symvar_value(opr.inputs[1])


class ConvolutionForwardOpr(MgeOpr):
    name = "ConvolutionForward"

    def __init__(self, opr):
        super().__init__(opr)
        self.kernel_shape = get_shape(opr.inputs[1])
        self.param_W = get_symvar_value(opr.inputs[1])
        self.data_format = self.params["format"]
        self.dilation_w = self.params["dilate_w"]
        self.dilation_h = self.params["dilate_h"]
        self.compute_mode = self.params["compute_mode"]
        self.sparse = self.params["sparse"]

        self.ph, self.pw = self.params["pad_h"], self.params["pad_w"]
        self.sh, self.sw = self.params["stride_h"], self.params["stride_w"]
        if self.data_format == "NCHW":
            self.kh = self.kernel_shape[-2]
            self.kw = self.kernel_shape[-1]
        else:
            assert False, "do not support this {} format".format(self.data_format)

        self.num_output = get_shape(opr.outputs[0])[1]
        self.bias_term = False
        self.group = self.param_W.shape[0] if self.param_W.ndim == 5 else 1


class ConvBiasForwardOpr(ConvolutionForwardOpr):
    name = "ConvBiasForward"

    def __init__(self, opr):
        super().__init__(opr)
        self.nonline_mode = self.params["nonlineMode"]


class ElemwiseOpr(MgeOpr):
    name = "Elemwise"

    def __init__(self, opr):
        super().__init__(opr)
        self.mode = self.params["mode"]


class ElemwiseMultiTypeOpr(MgeOpr):
    name = "ElemwiseMultiType"


class Host2DeviceCopyOpr(MgeOpr):
    name = "Host2DeviceCopy"

    def __init__(self, opr):
        super().__init__(opr)
        assert len(opr.outputs) == 1, "wrong number of outputs"
        self.shape = get_shape(opr.outputs[0])


class MultipleDeviceTensorHolderOpr(MgeOpr):
    name = "MultipleDeviceTensorHolder"


class VolatileSharedDeviceTensorOpr(MgeOpr):
    name = "VolatileSharedDeviceTensor"


class IndexingOneHotOpr(MgeOpr):
    name = "IndexingOneHotOpr"


class SubtensorOpr(MgeOpr):
    name = "Subtensor"

    def __init__(self, opr):
        super().__init__(opr)
        self.has_step = []
        self.has_begin = []
        self.has_end = []
        self.has_idx = []
        self.axis = []
        for param in self.params:
            self.has_step.append(param["step"])
            self.has_begin.append(param["begin"])
            self.has_end.append(param["end"])
            self.has_idx.append(param["idx"])
            self.axis.append(param["axis"])

        begin_param = []
        end_param = []
        step_param = []
        squeeze_axis = []
        slice_param = [get_symvar_value(v)[0] for v in opr.inputs[1:]]
        for i in range(len(self.has_begin)):
            if self.has_idx[i] == 1:
                begin_idx = slice_param.pop(0)
                end_idx = begin_idx + 1
                begin_param.append(begin_idx)
                end_param.append(end_idx)
                step_param.append(1)
                squeeze_axis.append(self.axis[i])
            else:
                if self.has_begin[i]:
                    begin_param.append(slice_param.pop(0))
                if self.has_end[i]:
                    end_param.append(slice_param.pop(0))
                step_param.append(1 if self.has_step[i] == 0 else slice_param.pop(0))

        self.squeeze_axis = squeeze_axis
        self.begin_param = begin_param
        self.end_param = end_param
        self.step_param = step_param


class ImmutableTensorOpr(MgeOpr):
    name = "ImmutableTensor"


class GetVarShapeOpr(MgeOpr):
    name = "GetVarShape"


class BatchNormForwardOpr(MgeOpr):
    name = "BatchNormForward"

    def __init__(self, opr):
        super().__init__(opr)
        self.output_idx = 4 if mge_version <= "0.6.0" else 2
        self.scale = get_symvar_value(opr.inputs[1]).squeeze()
        self.bias = get_symvar_value(opr.inputs[2]).squeeze()
        self.mean = get_symvar_value(opr.inputs[3]).squeeze()
        self.var = get_symvar_value(opr.inputs[4]).squeeze()


class MarkNoBroadcastElemwiseOpr(MgeOpr):
    name = "MarkNoBroadcastElemwise"

    def __init__(self, opr):
        super().__init__(opr)
        self.mode = "Identity"


class IdentityOpr(MgeOpr):
    name = "Identity"

    def __init__(self, opr):
        super().__init__(opr)
        self.mode = "Identity"


class SharedDeviceTensorOpr(MgeOpr):
    name = "SharedDeviceTensor"

    def __init__(self, opr):
        super().__init__(opr)
        assert len(opr.outputs) == 1, "wrong number of outputs"
        self.shape = get_shape(opr.outputs[0])


class DimshuffleOpr(MgeOpr):
    name = "Dimshuffle"

    def __init__(self, opr):
        super().__init__(opr)
        self.pattern = self.params["pattern"]
        self.ndim = self.params["ndim"]


class TypeCvtOpr(MgeOpr):
    name = "TypeCvt"


class ReduceOpr(MgeOpr):
    name = "Reduce"

    def __init__(self, opr):
        super().__init__(opr)
        self.axis = self.params["axis"]
        self.mode = self.params["mode"]


class AxisAddRemoveOpr(MgeOpr):
    name = "AxisAddRemove"

    def __init__(self, opr):
        super().__init__(opr)
        self.desc = self.params["desc"]
        self.nr_desc = self.params["nr_desc"]


class BroadcastOpr(MgeOpr):
    name = "Broadcast"


class WarpPerspectiveForwardOpr(MgeOpr):
    name = "WarpPerspectiveForward"


class LinspaceOpr(MgeOpr):
    name = "Linspace"


class ConvolutionBackwardDataOpr(MgeOpr):
    name = "ConvolutionBackwardData"

    def __init__(self, opr):
        super().__init__(opr)
        # opr.inputs[0] is conv kernel
        self.kernel_shape = get_shape(opr.inputs[0])
        self.param_W = get_symvar_value(opr.inputs[0])
        self.data_format = self.params["format"]
        self.dilation_w = self.params["dilate_w"]
        self.dilation_h = self.params["dilate_h"]
        self.compute_mode = self.params["compute_mode"]
        self.sparse = self.params["sparse"]

        self.ph, self.pw = self.params["pad_h"], self.params["pad_w"]
        self.sh, self.sw = self.params["stride_h"], self.params["stride_w"]
        if self.data_format == "NCHW":
            self.kh = self.kernel_shape[-2]
            self.kw = self.kernel_shape[-1]
        else:
            assert False, "do not support this {} format".format(self.data_format)

        self.num_output = get_shape(opr.outputs[0])[1]
        self.bias_term = False
        self.group = self.param_W.shape[0] if self.param_W.ndim == 5 else 1
