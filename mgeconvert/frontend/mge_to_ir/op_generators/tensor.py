# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ....converter_ir.ir_op import (
    AxisAddRemoveOpr,
    BroadcastOpr,
    ConcatOpr,
    ConstantOpr,
    GetSubTensorOpr,
    GetVarShapeOpr,
    IdentityOpr,
    MatMulOpr,
    MultipleDeviceTensorHolderOpr,
    ReduceOpr,
    ReshapeOpr,
    ResizeOpr,
    SharedDeviceTensorOpr,
    TransposeOpr,
    TypeCvtOpr,
    VolatileSharedDeviceTensorOpr,
)
from ..mge_utils import get_shape, get_symvar_value
from .base import OpGenBase, _register_op


@_register_op("ImmutableTensor")
class GenImmutableTensorOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.op = ConstantOpr()
        self.add_tensors(mge_opr)


@_register_op("MultipleDeviceTensorHolder")
class GenMultipleDeviceTensorHolderOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.op = MultipleDeviceTensorHolderOpr()
        self.add_tensors(mge_opr)


@_register_op("SharedDeviceTensorOpr")
class GenSharedDeviceTensorOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.op = SharedDeviceTensorOpr()
        self.add_tensors(mge_opr)


@_register_op("VolatileSharedDeviceTensor")
class GenVolatileSharedDeviceTensorOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.op = VolatileSharedDeviceTensorOpr()
        self.add_tensors(mge_opr)


@_register_op("Identity")
class GenIdentityOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.op = IdentityOpr()
        self.add_tensors(mge_opr)


@_register_op("GetVarShape")
class GenGetVarShapeOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.op = GetVarShapeOpr()
        self.add_tensors(mge_opr)
        self.op.out_tensors[0]._var = mge_opr.outputs[0]


@_register_op("Broadcast")
class GenBroadcastOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.op = BroadcastOpr()
        self.add_tensors(mge_opr)
        for i in range(len(self.op.out_tensors)):
            self.op.out_tensors[i]._var = mge_opr.outputs[i]


@_register_op("Concat")
class GenConcatOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.axis = self.params["axis"]
        self.op = ConcatOpr(self.axis)
        self.add_tensors(mge_opr)


@_register_op("Reshape")
class GenReshapeOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.output_shape = get_shape(mge_opr.outputs[0])

        self.op = ReshapeOpr(self.output_shape)
        self.add_tensors(mge_opr)


@_register_op("AxisAddRemove")
class GenAxisAddRemoveOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.desc = self.params["desc"]  # 0:add_axis, 1:remove_axis
        self.nr_desc = self.params["nr_desc"]
        self.output_shape = get_shape(mge_opr.outputs[0])
        self.op = AxisAddRemoveOpr(self.output_shape, self.desc)
        self.add_tensors(mge_opr)


@_register_op("Reduce")
class GenReduceOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.axis = self.params["axis"]
        self.mode = self.params["mode"]
        self.op = ReduceOpr(self.axis, self.mode, True)
        self.add_tensors(mge_opr)
        for i in range(len(self.op.out_tensors)):
            self.op.out_tensors[i]._var = mge_opr.outputs[i]


@_register_op("MatrixMul")
class GenMatrixMulOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.format = self.params["format"]
        self.transposeB = self.params["transposeB"]
        self.transposeA = self.params["transposeA"]
        self.compute_mode = self.params["compute_mode"]

        self.op = MatMulOpr(
            transpose_a=self.transposeA,
            transpose_b=self.transposeB,
            compute_mode=self.compute_mode,
            format=self.format,
        )
        self.add_tensors(mge_opr)


@_register_op("Dimshuffle")
class GenDimshuffleOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.pattern = self.params["pattern"]

        self.op = TransposeOpr(self.pattern)
        self.add_tensors(mge_opr)


@_register_op("Subtensor")
class GenSubtensorOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
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
        slice_param = [get_symvar_value(v)[0] for v in mge_opr.inputs[1:]]
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
                else:
                    begin_param.append(0)
                if self.has_end[i]:
                    end_param.append(slice_param.pop(0))
                else:
                    end_param.append(2147483647)
                step_param.append(1 if self.has_step[i] == 0 else slice_param.pop(0))

        self.op = GetSubTensorOpr(
            self.axis, begin_param, end_param, step_param, squeeze_axis=squeeze_axis
        )
        self.add_tensors(mge_opr)


@_register_op("TypeCvt")
class GenTypeCvtOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)

        out_dtype = mge_opr.outputs[0].dtype
        self.op = TypeCvtOpr(out_dtype)
        self.add_tensors(mge_opr)


@_register_op("ResizeForward")
class GenResizeOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.shape_param = get_symvar_value(mge_opr.inputs[1])
        self.op = ResizeOpr(self.shape_param)
        self.add_tensors(mge_opr)
