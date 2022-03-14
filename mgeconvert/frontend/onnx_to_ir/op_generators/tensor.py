# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import numpy as np

from ....converter_ir.ir_op import (
    ClipOpr,
    ConcatOpr,
    DropoutOpr,
    FlattenOpr,
    GatherOpr,
    GetSubTensorOpr,
    MatMulOpr,
    ReshapeOpr,
    ResizeOpr,
    TransposeOpr,
    TypeCvtOpr,
)
from ..onnxproto_resolver import onnx2np_dtype_mapping
from .base import OpGenBase, _register_op


@_register_op("Reshape")
class GenReshapeOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        super().__init__(node, ir_graph, resolver)
        self.op = ReshapeOpr([])
        self.add_tensors()

        if opset < 5:
            for attr in node.attribute:
                if attr.name == "shape":
                    out_shape = attr.ints
        else:
            ir_outshape_tensor = self.op.inp_tensors[1]
            out_shape = np.frombuffer(
                ir_outshape_tensor.np_data, ir_outshape_tensor.dtype
            )
            self.op.inp_tensors = [self.op.inp_tensors[0]]
            for attr in node.attribute:
                if attr.name == "allowzero":
                    self.op.allowzero = attr.i
        valid_test = [i for i, k in enumerate(out_shape) if k == -1]
        assert (
            len(valid_test) <= 1
        ), "Target Shape of ReShape Opr only contains '-1' Once At Most"

        self.op.out_shape = out_shape


@_register_op("Gemm")
class GenMatrixMulOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.alpha = 1.0
        self.beta = 1.0
        self.transA = 0
        self.transB = 0
        for attr in node.attribute:
            if attr.name == "alpha":
                self.alpha = attr.f
            elif attr.name == "beta":
                self.beta = attr.f
            elif attr.name == "transA":
                self.transA = attr.i
            elif attr.name == "transB":
                self.transB = attr.i
            else:
                raise Exception(f"Invalid Gemm attribute {attr.name}")

        self.op = MatMulOpr(
            transpose_a=self.transA,
            transpose_b=self.transB,
            alpha=self.alpha,
            beta=self.beta,
        )
        self.add_tensors()


@_register_op("Transpose")
class GenTransposeOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        assert node.attribute[0].name == "perm"
        self.pattern = node.attribute[0].ints
        self.op = TransposeOpr(self.pattern)
        self.add_tensors()


@_register_op("Slice")
class GenSliceOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        super().__init__(node, ir_graph, resolver)

        self.op = GetSubTensorOpr([], [], [], [])
        self.add_tensors()

        if opset == 1:
            axes = []
            for attr in node.attribute:
                if attr.name == "axes":
                    axes = np.array(attr.ints)
                elif attr.name == "starts":
                    starts = np.array(attr.ints)
                elif attr.name == "ends":
                    ends = np.array(attr.ints)

            rank = len(starts)
            if len(axes) == 0:
                axes = np.array(range(0, rank))
            assert len(ends) == rank
            steps = np.ones(rank)
        else:
            nr_inputs = len(self.op.inp_tensors)
            rank = len(self.op.inp_tensors[0].shape)
            axes = np.arange(0, rank)
            steps = np.ones(rank)

            ir_starts_tensor = self.op.inp_tensors[1]
            starts = np.frombuffer(ir_starts_tensor.np_data, ir_starts_tensor.dtype)

            ir_ends_tensor = self.op.inp_tensors[2]
            ends = np.frombuffer(ir_ends_tensor.np_data, ir_ends_tensor.dtype)

            if nr_inputs >= 4:
                ir_axes_tensor = self.op.inp_tensors[3]
                axes = np.frombuffer(ir_axes_tensor.np_data, ir_axes_tensor.dtype)

            if nr_inputs == 5:
                ir_step_tensor = self.op.inp_tensors[4]
                steps = np.frombuffer(ir_step_tensor.np_data, ir_step_tensor.dtype)

            self.op.inp_tensors = [self.op.inp_tensors[0]]
            assert len(starts) == len(ends)
            assert len(starts) == len(axes)
            assert len(starts) == len(steps)

        self.op.axis = axes
        self.op.begin_params = starts
        self.op.end_params = ends
        self.op.step_params = steps


@_register_op("Cast")
class GenCastOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        assert node.attribute[0].name == "to"
        out_dtype = onnx2np_dtype_mapping[node.attribute[0].i]
        self.op = TypeCvtOpr(out_dtype)
        self.add_tensors()


@_register_op("Flatten")
class GenFlattenOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        assert node.attribute[0].name == "axis"
        axis = node.attribute[0].i
        self.op = FlattenOpr(axis)
        self.add_tensors()


@_register_op("Clip")
class GenClipOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        super().__init__(node, ir_graph, resolver)

        max = float("inf")
        min = float("-inf")
        self.op = ClipOpr(max, min)
        self.add_tensors()

        if opset < 11:
            for attr in node.attribute:
                if attr.name == "max":
                    max = attr.f
                elif attr.name == "min":
                    min = attr.f
                else:
                    raise AssertionError("Invalid attribute in Clip")
        else:
            nr_inputs = len(node.input)
            assert nr_inputs in (1, 3), "NR Inputs of Clip must be 1 or 3"
            if nr_inputs == 3:
                min_tensor = self.op.inp_tensors[1]
                max_tensor = self.op.inp_tensors[2]
                min = np.frombuffer(min_tensor.np_data, dtype=min_tensor.dtype)[0]
                max = np.frombuffer(max_tensor.np_data, dtype=max_tensor.dtype)[0]
                self.op.inp_tensors = [self.op.inp_tensors[0]]

        self.op.upper = max
        self.op.lower = min


@_register_op("Concat")
class GenConcatOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        assert node.attribute[0].name == "axis"
        axis = node.attribute[0].i
        self.op = ConcatOpr(axis)
        self.add_tensors()


@_register_op("Dropout")
class GenDropoutOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        super().__init__(node, ir_graph, resolver)

        ratio = 0.5
        training_mode = False
        self.op = DropoutOpr(ratio, training_mode)
        self.add_tensors()

        if opset < 12:
            for attr in node.attribute:
                if attr.name == "is_test":
                    training_mode = attr.i == 0
                elif attr.name == "ratio":
                    ratio = attr.f
        else:
            nr_inps = len(self.op.inp_tensors)
            if nr_inps >= 2:
                ratio = np.frombuffer(
                    self.op.inp_tensors[1].np_data, dtype=self.op.inp_tensors[1].dtype
                )
            if nr_inps == 3:
                training_mode = np.frombuffer(
                    self.op.inp_tensors[2].np_data, dtype=self.op.inp_tensors[2].dtype
                )
            self.op.inp_tensors = [self.op.inp_tensors[0]]

        self.op.ratio_prob = ratio[0]
        self.op.training = training_mode[0]


@_register_op("Resize")
class GenResizeOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        self.op = ResizeOpr(-1)
        self.add_tensors()

        param = {
            "mode": "nearest",
            "coordinate_transformation_mode": "half_pixel",
            "cubic_coeff_a": -0.75,
            "exclude_outside": 0,
            "extrapolation_value": 0.0,
            "nearest_mode": "round_prefer_floor",
        }

        for attr in node.attribute:
            name = attr.name
            if name in ("mode", "coordinate_transformation_mode", "nearest_mode"):
                param[name] = str(attr.s, "utf-8")
            elif name in ("cubic_coeff_a", "extrapolation_value"):
                param[name] = attr.f
            else:
                assert name == "exclude_outside"
                param[name] = attr.i

        nr_inp_tensors = len(self.op.inp_tensors)
        if nr_inp_tensors == 2:
            param["scale"] = np.frombuffer(
                self.op.inp_tensors[1].np_data, self.op.inp_tensors[1].dtype
            )
        else:
            for i in self.op.inp_tensors[1:]:
                np_data = (
                    np.frombuffer(i.np_data, i.dtype) if i.np_data is not None else None
                )
                if i.dtype == np.int64:
                    param["sizes"] = np_data
                else:
                    if np_data is not None and len(np_data) == len(
                        self.op.inp_tensors[0].shape
                    ):
                        param["scale"] = np_data
                    else:
                        param["roi"] = np_data

        self.op.inp_tensors = [self.op.inp_tensors[0]]
        self.op.extra_param = param


@_register_op("Gather")
class GenGatherOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        axis = 0
        self.op = GatherOpr(axis)
        self.add_tensors()

        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i

        self.op.axis = axis
