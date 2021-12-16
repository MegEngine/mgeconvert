# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ....converter_ir.ir_op import AvgPool2dOpr, Conv2dOpr, MaxPool2dOpr
from .base import OpGenBase, _register_op


@_register_op("Conv")
class GenConv2dOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver):
        super().__init__(node, ir_graph, resolver)
        stride = (1, 1)
        padding = (0, 0)
        dilation = (1, 1)
        groups = 1
        auto_pad = "NOTSET"
        for attr in node.attribute:
            if attr.name == "strides":
                stride = tuple(attr.ints)
            elif attr.name == "pads":
                padding = attr.ints
            elif attr.name == "dilations":
                dilation = tuple(attr.ints)
            elif attr.name == "group":
                groups = attr.i
            elif attr.name == "auto_pad":
                auto_pad = attr.strings

        self.op = Conv2dOpr(stride, padding, dilation, groups, auto_pad)
        self.add_tensors()


@_register_op("MaxPool", "GlobalMaxPool", "GlobalAveragePool", "AveragePool")
class GenMaxPoolOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver):
        super().__init__(node, ir_graph, resolver)
        auto_pad = "NOTSET"
        ceil_mode = 0
        strides = (1, 1)
        pads = (0, 0)
        kernel_shape = None

        dilations = (1, 1)
        storage_order = 0
        count_include_pad = 0

        for attr in node.attribute:
            if attr.name == "strides":
                strides = tuple(attr.ints)
            elif attr.name == "pads":
                pads = attr.ints
            elif attr.name == "dilations":
                dilations = tuple(attr.ints)
            elif attr.name == "ceil_mode":
                ceil_mode = attr.i
            elif attr.name == "kernel_shape":
                kernel_shape = tuple(attr.ints)
            elif attr.name == "auto_pad":
                auto_pad = attr.strings
            elif attr.name == "storage_order":
                storage_order = attr.i
            elif attr.name == "count_include_pad":
                count_include_pad = attr.i

        if node.op_type == "MaxPool":
            self.op = MaxPool2dOpr(
                kernel_shape,
                strides,
                pads,
                auto_pad,
                ceil_mode,
                dilations,
                storage_order,
            )
        elif node.op_type == "GlobalMaxPool":
            self.op = MaxPool2dOpr((-1, -1), (1, 1), (0, 0))
        elif node.op_type == "AveragePool":
            mode_map = {
                0: "average_count_exclude_padding",
                1: "average_count_include_padding",
            }
            self.op = AvgPool2dOpr(
                kernel_shape,
                strides,
                pads,
                mode_map[count_include_pad],
                auto_pad,
                ceil_mode,
            )
        elif node.op_type == "GlobalAveragePool":
            self.op = AvgPool2dOpr((-1, -1), (1, 1), (0, 0))

        self.add_tensors()
        if "Global" in node.op_type:
            shape = self.op.inp_tensors[0].shape
            self.op.kernel_size = [shape[2], shape[3]]
