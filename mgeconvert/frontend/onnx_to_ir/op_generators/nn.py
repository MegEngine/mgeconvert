# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from ....converter_ir.ir_op import AvgPool2dOpr, Conv2dOpr, LstmOpr, MaxPool2dOpr
from .base import OpGenBase, _register_op


@_register_op("Conv")
class GenConv2dOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
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
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
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


@_register_op("LSTM")
class GenLSTMOpr(OpGenBase):
    def __init__(self, node, ir_graph, resolver, opset):
        # pylint: disable=W0612,W0613
        super().__init__(node, ir_graph, resolver)

        activation_alpha = None
        activation_beta = None
        activations = None
        clip = None
        direction = "forward"
        hidden_size = 16
        input_forget = None
        output_sequence = None
        layout = 0

        for attr in node.attribute:
            if attr.name == "activation_alpha":
                activation_alpha = attr.floats
            if attr.name == "activation_beta":
                activation_beta = attr.floats
            if attr.name == "activations":
                activations = attr.strings
            if attr.name == "clip":
                clip = attr.f
            if attr.name == "direction":
                direction = str(attr.s, "utf-8")
            if attr.name == "hidden_size":
                hidden_size = attr.i
            if attr.name == "input_forget":
                input_forget = attr.i
            if attr.name == "output_sequence":
                output_sequence = attr.i
            if attr.name == "layout":
                layout = attr.i

        self.op = LstmOpr(0, 0, 1, False, False, 0, direction, 0)
        self.add_tensors()
        inp_tensors = self.op.inp_tensors

        # input_size
        self.op.input_size = inp_tensors[0].shape[2]
        self.op.batch_size = inp_tensors[0].shape[1]
        self.op.hidden_size = hidden_size
        if layout:
            self.op.batch_first = True

        W = np.frombuffer(inp_tensors[1].np_data, dtype=inp_tensors[1].dtype).reshape(
            inp_tensors[1].shape
        )
        R = np.frombuffer(inp_tensors[2].np_data, dtype=inp_tensors[2].dtype).reshape(
            inp_tensors[2].shape
        )
        # weight_ih
        self.op.weight_ih_l.append(
            np.concatenate(
                (
                    W[0][:hidden_size],
                    W[0][2 * hidden_size : 4 * hidden_size],
                    W[0][hidden_size : 2 * hidden_size],
                ),
                axis=0,
            )
        )
        # weight_hh
        self.op.weight_hh_l.append(
            np.concatenate(
                (
                    R[0][:hidden_size],
                    R[0][2 * hidden_size : 4 * hidden_size],
                    R[0][hidden_size : 2 * hidden_size],
                ),
                axis=0,
            )
        )
        # bias_ih and bias_hh
        try:
            B = np.frombuffer(
                inp_tensors[3].np_data, dtype=inp_tensors[3].dtype
            ).reshape(inp_tensors[3].shape)
            self.op.bias_ih_l.append(
                np.concatenate(
                    (
                        B[0][:hidden_size],
                        B[0][2 * hidden_size : 4 * hidden_size],
                        B[0][hidden_size : 2 * hidden_size],
                    ),
                    axis=0,
                )
            )
            self.op.bias_hh_l.append(
                np.concatenate(
                    (
                        B[0][4 * hidden_size : 5 * hidden_size],
                        B[0][6 * hidden_size : 8 * hidden_size],
                        B[0][5 * hidden_size : 6 * hidden_size],
                    ),
                    axis=0,
                )
            )
            self.op.bias = True
        except IndexError:
            pass

        if direction == "bidirectional":
            self.op.weight_ih_l_reverse.append(
                np.concatenate(
                    (
                        W[1][:hidden_size],
                        W[1][2 * hidden_size : 4 * hidden_size],
                        W[1][hidden_size : 2 * hidden_size],
                    ),
                    axis=0,
                )
            )
            self.op.weight_hh_l_reverse.append(
                np.concatenate(
                    (
                        R[1][:hidden_size],
                        R[1][2 * hidden_size : 4 * hidden_size],
                        R[1][hidden_size : 2 * hidden_size],
                    ),
                    axis=0,
                )
            )
            try:
                self.op.bias_ih_l_reverse.append(
                    np.concatenate(
                        (
                            B[1][:hidden_size],
                            B[1][2 * hidden_size : 4 * hidden_size],
                            B[1][hidden_size : 2 * hidden_size],
                        ),
                        axis=0,
                    )
                )
                self.op.bias_hh_l_reverse.append(
                    np.concatenate(
                        (
                            B[1][4 * hidden_size : 5 * hidden_size],
                            B[1][6 * hidden_size : 8 * hidden_size],
                            B[1][5 * hidden_size : 6 * hidden_size],
                        ),
                        axis=0,
                    )
                )
            except NameError:
                pass

        # sequence_lens
        try:
            self.op.sequence_lens = inp_tensors[4].np_data
        except IndexError:
            pass

        inps = [inp_tensors[0]]

        # init_h
        try:
            inps.append(inp_tensors[5])
        except IndexError:
            pass

        # init_c
        try:
            inps.append(inp_tensors[6])
        except IndexError:
            pass

        # p
        try:
            self.op.p = inp_tensors[7].np_data
        except IndexError:
            pass

        self.op.activation_alpha = activation_alpha
        self.op.activation_beta = activation_beta
        self.op.activations = activations
        self.op.clip = clip
        self.op.input_forget = input_forget
        self.op.output_sequence = output_sequence

        self.op.inp_tensors = inps
