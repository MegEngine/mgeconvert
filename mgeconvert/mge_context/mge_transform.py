# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# type: ignore[no-redef]

from collections import OrderedDict
from enum import Enum
from typing import Callable, Dict, List, Sequence

import numpy as np

from . import mge_op as Ops
from .mge_op import (
    ConvBiasForwardOpr,
    ConvForwardBiasOpr,
    ConvolutionBackwardDataOpr,
    ConvolutionForwardOpr,
    DimshuffleOpr,
    ElemwiseOpr,
    OpBase,
    PadOpr,
    PoolingForwardOpr,
    ReduceOpr,
    Relu6Opr,
    ReshapeOpr,
    SoftmaxOpr,
    TypeCvtOpr,
)
from .mge_tensor import FakeSymbolVar, Tensor
from .mge_utils import isconst


class PatternNode:
    def __init__(self, type, is_output=False, const_value=None):
        self.op = None
        self.type = type
        self.inp_oprs = []
        self.inp_const = []
        self.inp_vars = []
        self.is_output = is_output
        self.const_value = const_value

    def check_const_value(self, op):
        inp_vars = [v.np_data for v in op.inp_vars]
        for const in self.const_value:
            idx = const[0]
            if idx == -1:
                find = False
                for index, v in enumerate(inp_vars):
                    if np.array_equal(const[1], v):
                        find = True
                        del inp_vars[index]
                        break
                if not find:
                    return False
            elif not np.array_equal(const[1], inp_vars[idx]):
                return False
        return True


get_type = lambda op: op.mode if isinstance(op, Ops.ElemwiseOpr) else type(op).__name__


def match(node, opr):
    node_queue = [node]
    opr_queue = [opr]
    matched_opr = set()
    matched_node = set()
    while len(node_queue) != 0:
        cur_node = node_queue.pop(0)
        cur_opr = opr_queue.pop(0)
        if cur_node.type != get_type(cur_opr) and cur_node.type != "*" or cur_opr.skip:
            return False
        if cur_node.op == None:
            cur_node.op = cur_opr
            if cur_node.const_value != None:
                if not cur_node.check_const_value(cur_opr):
                    return False
        elif cur_node.op != cur_opr:
            return False

        matched_opr.add(cur_opr)
        matched_node.add(cur_node)
        for i, var in enumerate(cur_opr.inp_vars):
            if isconst(var):
                cur_node.inp_const.append([i, var.np_data])
            else:
                cur_node.inp_vars.append([i, var])
        if len(cur_node.inp_oprs) == 0:
            continue
        if len(cur_node.inp_oprs) != len(cur_opr.inp_oprs):
            return False

        for i, j in zip(cur_node.inp_oprs, cur_opr.inp_oprs):
            node_queue.append(i)
            opr_queue.append(j)

    for n in matched_node:
        if n.is_output:
            continue
        for op in n.op.out_oprs:
            if op not in matched_opr:
                return False

    return True


class TransformerRule(Enum):
    # general rules
    NOPE = 1

    # for TFLite
    REDUCE_AXIS_AS_INPUT = 100
    REMOVE_RESHAPE_INPUT = 101
    # FUSE_FOR_RELU6 pass should happen before FUSE_ACTIVATION
    FUSE_FOR_RELU6 = 102
    FUSE_ACTIVATION = 103
    CONV_ADD_ZERO_BIAS = 104
    DEPTHWISE_CONV_RESHAPE_WEIGHT = 105
    FUSE_SOFTMAX = 106
    DECONV_SHAPE_AS_INPUT = 107
    FUSE_ASTYPE = 108
    PADDING_FOR_CONV = 109
    TRANSPOSE_PATTERN_AS_INPUT = 110
    # FUSE_FOR_LEAKY_RELU should happen before EXPAND_MUL_ADD3
    FUSE_FOR_LEAKY_RELU = 111
    EXPAND_MUL_ADD3 = 112
    EXPAND_ADD_SIGMOID = 113
    FUSE_FOR_CONV_BIAS = 114
    FUSE_FOR_DECONV_BIAS = 115
    FUSE_FOR_FULLY_CONNECTED = 116
    RESHAPE_BIAS_TO_1DIM = 117


TRANSFORMMAP: Dict[Enum, Callable] = {}


def optimize_for_conversion(net, transformer_options):
    if not isinstance(transformer_options, Sequence):
        transformer_options = (transformer_options,)
    for option in transformer_options:
        TRANSFORMMAP[option](net)


def _register_tranformation_rule(transformer_option):
    def callback(impl):
        TRANSFORMMAP[transformer_option] = impl

    return callback


@_register_tranformation_rule(TransformerRule.REMOVE_RESHAPE_INPUT)
def _remove_reshape_input(net):
    for op in net.all_oprs:
        if not isinstance(op, ReshapeOpr):
            continue

        del op.inp_vars[1]


@_register_tranformation_rule(TransformerRule.REDUCE_AXIS_AS_INPUT)
def _reduce_axis_as_input(net):
    for op in net.all_oprs:
        if not isinstance(op, ReduceOpr):
            continue

        byte_list = np.int32(op.axis).tobytes()

        axis_symvar = FakeSymbolVar(
            sid=net.max_id,
            name=op.name + "_axis",
            shape=[1],
            dtype=np.int32,
            owner=op,
            byte_list=byte_list,
        )
        net.max_id += 1

        axis_tensor = net.get_var(axis_symvar)
        op.add_inp_var(axis_tensor)


@_register_tranformation_rule(TransformerRule.FUSE_FOR_RELU6)
def _fuse_for_relu6(net):
    matches = OrderedDict()

    for op in net.all_oprs:
        if not isinstance(op, ElemwiseOpr):
            continue
        if len(op.inp_oprs) <= 0 or len(op.inp_oprs) >= 2:
            continue
        prev_op = op.inp_oprs[0]
        cur_index = net._opr_ids.index(op.id)

        if op.mode == "MIN" and np.array_equal(op.inp_vars[1].np_data, np.array([6])):
            if (
                isinstance(prev_op, ElemwiseOpr)
                and prev_op.mode == "MAX"
                and np.array_equal(prev_op.inp_vars[1].np_data, np.array([0]))
                and net._opr_ids.index(prev_op.id) == cur_index - 1
            ):
                relu6_opr = Relu6Opr()
                relu6_opr.inp_vars = prev_op.inp_vars
                relu6_opr.out_vars = op.out_vars
                relu6_opr.inp_oprs = [prev_op.prev_opr]
                relu6_opr.out_oprs = op.out_oprs
                matches[prev_op.id] = (net.max_id, relu6_opr)
                net.max_id += 1
        if op.mode == "MAX" and np.array_equal(op.inp_vars[1].np_data, np.array([0])):
            if (
                isinstance(prev_op, ElemwiseOpr)
                and prev_op.mode == "MIN"
                and np.array_equal(prev_op.inp_vars[1].np_data, np.array([6]))
                and net._opr_ids.index(prev_op.id) == cur_index - 1
            ):
                relu6_opr = Relu6Opr()
                relu6_opr.inp_vars = prev_op.inp_vars
                relu6_opr.out_vars = op.out_vars
                relu6_opr.inp_oprs = [prev_op.inp_oprs[0]]
                relu6_opr.out_oprs = op.out_oprs
                matches[prev_op.id] = (net.max_id, relu6_opr)
                net.max_id += 1

    for original_id, generated_pair in list(matches.items())[::-1]:
        index = net._opr_ids.index(original_id)
        del net._opr_ids[index : index + 2]
        del net.all_oprs[index : index + 2]

        net._opr_ids.insert(index, generated_pair[0])
        net.all_oprs.insert(index, generated_pair[1])


@_register_tranformation_rule(TransformerRule.FUSE_ACTIVATION)
def _fuse_activation(net):
    delete_intended = []

    for op_id, op in zip(net._opr_ids, net.all_oprs):
        if isinstance(op, Relu6Opr) or (
            isinstance(op, ElemwiseOpr) and op.mode in ("RELU", "TANH")
        ):
            prev_op = op.inp_oprs[0]
            if prev_op.activation != "IDENTITY":
                continue

            # activation(relu/relu6/tanh) must be fused with previous opr
            activation = getattr(op, "mode", "IDENTITY")
            activation = "RELU6" if isinstance(op, Relu6Opr) else activation
            prev_op.activation = activation
            prev_op.out_vars = op.out_vars

            for post_op in op.out_oprs:
                idx = post_op.inp_oprs.index(op)
                post_op.inp_oprs[idx] = prev_op
                if post_op not in prev_op.out_oprs:
                    prev_op.out_oprs.append(post_op)

            delete_intended.append(net._opr_ids.index(op_id))

    for delete_idx in delete_intended[::-1]:
        del net._opr_ids[delete_idx]
        del net.all_oprs[delete_idx]


@_register_tranformation_rule(TransformerRule.CONV_ADD_ZERO_BIAS)
def _conv_add_zero_bias(net):
    for op in net.all_oprs:
        if not isinstance(op, ConvolutionForwardOpr):
            continue
        if isinstance(op, (ConvBiasForwardOpr, ConvForwardBiasOpr)):
            continue

        result_shape = [op.out_vars[0].shape[1]]
        if op.inp_vars[0].dtype == np.float32:
            dtype = np.float32
        else:  # op.inp_vars[0].dtype == np.uint8
            scale0 = op.inp_vars[0].dtype.metadata["mgb_dtype"]["scale"]
            scale1 = op.inp_vars[1].dtype.metadata["mgb_dtype"]["scale"]
            dtype = np.dtype(
                "int32",
                metadata={
                    "mgb_dtype": {"name": "Quantized8Asymm", "scale": scale0 * scale1}
                },
            )

        byte_list = np.zeros(
            result_shape, np.int32 if dtype == np.int32 else np.float32
        ).tobytes()
        bias_symvar = FakeSymbolVar(
            sid=net.max_id,
            name=op.name + "_bias",
            shape=result_shape,
            dtype=dtype,
            owner=op,
            byte_list=byte_list,
        )
        net.max_id += 1

        bias_tensor = net.get_var(bias_symvar)
        op.add_inp_var(bias_tensor)


@_register_tranformation_rule(TransformerRule.DEPTHWISE_CONV_RESHAPE_WEIGHT)
def _depthwise_conv_reshape_weight(net):
    # general group conv is not supported for TFLite
    for op in net.all_oprs:
        if not isinstance(op, ConvolutionForwardOpr):
            continue
        if op.group == 1:
            continue

        var = op.inp_vars[1]
        ic, cm = var.shape[1], var.shape[2] * op.group
        h, w = var.shape[3:5]
        var.shape = (ic, cm, h, w)
        var.ndim = len(var.shape)
        var.np_data.reshape(ic, cm, h, w)


@_register_tranformation_rule(TransformerRule.FUSE_SOFTMAX)
def _fuse_softmax(net):
    matches = OrderedDict()

    for op in net.all_oprs:
        if not isinstance(op, ElemwiseOpr) or op.mode != "TRUE_DIV":
            continue
        try:
            prev_op = op.inp_oprs[1]
            cur_index = net._opr_ids.index(op.id)
            if (
                not isinstance(prev_op, ReduceOpr)
                or prev_op.mode != "SUM"
                or prev_op.axis != 1
                or net._opr_ids.index(prev_op.id) != cur_index - 1
            ):
                continue
            prev_op = op.inp_oprs[0]
            if (
                not isinstance(prev_op, ElemwiseOpr)
                or prev_op.mode != "EXP"
                or net._opr_ids.index(prev_op.id) != cur_index - 2
            ):
                continue
            prev_op = prev_op.prev_opr
            if (
                not isinstance(prev_op, ElemwiseOpr)
                or prev_op.mode != "SUB"
                or net._opr_ids.index(prev_op.id) != cur_index - 3
            ):
                continue
            prev_op = prev_op.inp_oprs[1]
            if (
                not isinstance(prev_op, ReduceOpr)
                or prev_op.mode != "MAX"
                or prev_op.axis != 1
                or net._opr_ids.index(prev_op.id) != cur_index - 4
            ):
                continue
        except IndexError:  # doesn't match
            continue

        softmax_opr = SoftmaxOpr()
        softmax_opr.beta = 1
        softmax_opr.inp_vars = prev_op.inp_vars[:1]
        softmax_opr.out_vars = op.out_vars
        softmax_opr.inp_oprs = [prev_op.inp_oprs[0]]
        softmax_opr.out_oprs = op.out_oprs
        softmax_opr.prev_opr = prev_op.inp_oprs[0]
        matches[prev_op.id] = (net.max_id, softmax_opr)
        net.max_id += 1

    for original_id, generated_pair in list(matches.items())[::-1]:
        index = net._opr_ids.index(original_id)
        del net._opr_ids[index : index + 5]
        del net.all_oprs[index : index + 5]

        net._opr_ids.insert(index, generated_pair[0])
        net.all_oprs.insert(index, generated_pair[1])


@_register_tranformation_rule(TransformerRule.DECONV_SHAPE_AS_INPUT)
def _deconv_shape_as_input(net):
    for op in net.all_oprs:
        if not isinstance(op, ConvolutionBackwardDataOpr):
            continue

        byte_list = []
        result_shape = op.out_vars[0].shape
        number_list = [
            result_shape[0],
            result_shape[2],
            result_shape[3],
            result_shape[1],
        ]
        for i in number_list:
            byte_list.extend(np.int32(i).tobytes())
        shape_symvar = FakeSymbolVar(
            sid=net.max_id,
            name=op.name + "_shape",
            shape=[4],
            dtype=np.int32,
            owner=op,
            byte_list=byte_list,
        )
        net.max_id += 1
        shape_tensor = net.get_var(shape_symvar)
        op.inp_vars = [shape_tensor, op.inp_vars[1], op.inp_vars[0]]


@_register_tranformation_rule(TransformerRule.PADDING_FOR_CONV)
def _make_padding(net):
    def have_padding(opr):
        if (
            hasattr(opr, "ph")
            and (opr.ph > 0 or opr.pw > 0)
            and not isinstance(opr, ConvolutionBackwardDataOpr)
        ):
            return True
        return False

    insert_intended = OrderedDict()

    for op in net.all_oprs:
        if type(op) not in (  # pylint: disable=unidiomatic-typecheck
            ConvolutionForwardOpr,
            ConvBiasForwardOpr,
            PoolingForwardOpr,
        ):
            continue

        if have_padding(op):
            assert op.inp_vars[0].ndim == 4, "ERROR: unsupported padding mode"
            byte_list = []
            number_list = [0, 0, op.ph, op.ph, op.pw, op.pw, 0, 0]
            for i in number_list:
                byte_list.extend(np.int32(i).tobytes())
            pad_in_symvar = FakeSymbolVar(
                sid=net.max_id,
                name=op.name + "_pad_in",
                shape=[4, 2],
                dtype=np.int32,
                owner=None,
                byte_list=byte_list,
            )
            net.max_id += 1
            net._var_ids.append(pad_in_symvar.id)
            pad_in_tensor = Tensor(pad_in_symvar, None)
            net.all_vars.append(pad_in_tensor)

            shape = list(op.inp_vars[0].shape)
            pad_out_symvar = FakeSymbolVar(
                sid=net.max_id,
                name=op.name + "_pad_out",
                shape=[shape[0], shape[2] + op.ph * 2, shape[3] + op.pw * 2, shape[1],],
                dtype=op.inp_vars[0].dtype,
                owner=None,
                byte_list=None,
            )
            net.max_id += 1
            net._var_ids.append(pad_out_symvar.id)
            pad_out_tensor = Tensor(pad_out_symvar, None)
            net.all_vars.append(pad_out_tensor)

            pad_opr = PadOpr()
            pad_opr.type = "PadOpr"
            pad_opr.name = "pad_" + op.name
            pad_opr.id = net.max_id
            net.max_id += 1
            pad_opr.inp_vars = [op.inp_vars[0], pad_in_tensor]
            pad_opr.out_vars = [pad_out_tensor]
            pad_opr.inp_oprs = [op.inp_oprs[0]]
            pad_opr.out_oprs = [op]
            pad_out_tensor.owner = pad_opr
            op.inp_vars = [pad_out_tensor] + op.inp_vars[1:]
            op.inp_oprs = [pad_opr] + op.inp_oprs[1:]

            index = net._opr_ids.index(op.id)
            insert_intended[index] = (pad_opr.id, pad_opr)

    for index, generated_pair in list(insert_intended.items())[::-1]:
        net._opr_ids.insert(index, generated_pair[0])
        net.all_oprs.insert(index, generated_pair[1])


@_register_tranformation_rule(TransformerRule.FUSE_ASTYPE)
def _fuse_astype(net):
    def check_dtype(opr, dtype1, dtype2):
        if opr.inp_vars[0].dtype == dtype1 and opr.out_vars[0].dtype == dtype2:
            return True
        return False

    delete_intended = []

    for op_id, op in zip(net._opr_ids, net.all_oprs):
        if isinstance(op, TypeCvtOpr):
            # typecvt.prev_opr must have single output
            prev_op = op.prev_opr
            if (
                check_dtype(op, np.float32, np.int32)
                or check_dtype(op, np.float32, np.uint8)
                or check_dtype(op, np.float32, np.int8)
                or check_dtype(op, np.int32, np.int8)
            ):  # quant phase
                is_net_input = prev_op.out_vars[0] in net.input_vars
                if is_net_input:
                    net.input_vars.remove(prev_op.out_vars[0])
                prev_op.out_vars[0] = op.out_vars[0]
                op.out_vars[0].owner = prev_op
                if is_net_input:
                    net.input_vars.append(prev_op.out_vars[0])
                delete_intended.append(net._opr_ids.index(op_id))
            else:  # dequant phase, typecvt must be the last opr in the model
                if (
                    check_dtype(op, np.int8, np.int32)
                    or check_dtype(op, np.uint8, np.float32)
                    or check_dtype(op, np.int8, np.float32)
                ):
                    is_net_output = op.out_vars[0] in net.output_vars
                    if is_net_output:
                        net.output_vars.remove(op.out_vars[0])
                        net.output_vars.append(prev_op.out_vars[0])
                    delete_intended.append(net._opr_ids.index(op_id))

    for delete_idx in delete_intended[::-1]:
        del net._opr_ids[delete_idx]
        del net.all_oprs[delete_idx]


@_register_tranformation_rule(TransformerRule.TRANSPOSE_PATTERN_AS_INPUT)
def _transpose_pattern_as_input(net):
    for op in net.all_oprs:
        if not isinstance(op, DimshuffleOpr):
            continue

        pat = op.pattern
        assert len(pat) == 4, "only 4D input is supported by Dimshuffle"
        # NCHW perm -> NHWC perm, (0, 3, 1, 2) is inv(0, 2, 3, 1)
        pat = [pat[0], pat[3], pat[1], pat[2]]
        byte_list = []
        for i in pat:
            byte_list.extend(np.int32(i).tobytes())

        pattern_symvar = FakeSymbolVar(
            sid=net.max_id,
            name=op.name + "_pattern",
            shape=[len(pat)],
            dtype=np.int32,
            owner=op,
            byte_list=byte_list,
        )
        net.max_id += 1

        pattern_tensor = net.get_var(pattern_symvar)
        op.add_inp_var(pattern_tensor)


@_register_tranformation_rule(TransformerRule.EXPAND_MUL_ADD3)
def _expand_mul_add3(net):
    insert_intended = OrderedDict()

    for op in net.all_oprs:
        if not isinstance(op, ElemwiseOpr):
            continue
        if op.mode != "FUSE_MUL_ADD3":
            continue

        mul_out_symvar = FakeSymbolVar(
            sid=net.max_id,
            name=op.name + "_mul_out",
            shape=op.inp_vars[0].shape,
            dtype=op.inp_vars[0].dtype,
            owner=None,
            byte_list=None,
        )
        net.max_id += 1
        net._var_ids.append(mul_out_symvar.id)
        mul_out_tensor = Tensor(mul_out_symvar, None)
        net.all_vars.append(mul_out_tensor)

        mul_config = OpBase()
        mul_config.name = "mul_" + op.name
        mul_config.id = net.max_id
        mul_config.params = '{"mode": "MUL"}'
        mul_opr = ElemwiseOpr(mul_config)
        mul_opr.type = "ElemwiseOpr"
        net.max_id += 1
        mul_out_tensor.owner = mul_opr
        mul_opr.inp_vars = op.inp_vars[:2]
        mul_opr.out_vars = [mul_out_tensor]
        mul_opr.inp_oprs = op.inp_oprs[:2]
        mul_opr.out_oprs = [op]

        op.mode = "ADD"
        op.inp_vars = [mul_out_tensor, op.inp_vars[2]]
        op.inp_oprs = [mul_opr]
        if len(op.inp_oprs) > 2:
            op.inp_oprs.append(op.inp_oprs[2])

        index = net._opr_ids.index(op.id)
        insert_intended[index] = (mul_opr.id, mul_opr)

    for index, generated_pair in list(insert_intended.items())[::-1]:
        net._opr_ids.insert(index, generated_pair[0])
        net.all_oprs.insert(index, generated_pair[1])


@_register_tranformation_rule(TransformerRule.EXPAND_ADD_SIGMOID)
def _expand_add_sigmoid(net):
    insert_intended = OrderedDict()

    for op in net.all_oprs:
        if not isinstance(op, ElemwiseOpr):
            continue
        if op.mode != "FUSE_ADD_SIGMOID":
            continue

        add_out_symvar = FakeSymbolVar(
            sid=net.max_id,
            name=op.name + "_add_out",
            shape=op.inp_vars[0].shape,
            dtype=op.inp_vars[0].dtype,
            owner=None,
            byte_list=None,
        )
        net.max_id += 1
        net._var_ids.append(add_out_symvar.id)
        add_out_tensor = Tensor(add_out_symvar, None)
        net.all_vars.append(add_out_tensor)

        sigmoid_config = OpBase()
        sigmoid_config.name = "sigmoid_" + op.name
        sigmoid_config.id = net.max_id
        sigmoid_config.params = '{"mode": "SIGMOID"}'
        sigmoid_opr = ElemwiseOpr(sigmoid_config)
        sigmoid_opr.type = "ElemwiseOpr"
        net.max_id += 1

        sigmoid_opr.inp_vars = [add_out_tensor]
        sigmoid_opr.out_vars = op.out_vars
        sigmoid_opr.inp_oprs = [op]
        sigmoid_opr.out_oprs = op.out_oprs

        add_out_tensor.owner = op
        op.mode = "ADD"
        op.out_vars = [add_out_tensor]
        op.out_oprs = [sigmoid_opr]

        index = net._opr_ids.index(op.id) + 1
        insert_intended[index] = (sigmoid_opr.id, sigmoid_opr)

    for index, generated_pair in list(insert_intended.items())[::-1]:
        net._opr_ids.insert(index, generated_pair[0])
        net.all_oprs.insert(index, generated_pair[1])


def _fuse_for_leaky_relu(opr):
    assert (
        len(opr.out_oprs) == 1
        and isinstance(opr.out_oprs[0], Ops.ElemwiseOpr)
        and opr.out_oprs[0].mode == "ADD"
    )
    add_node = PatternNode("ADD", is_output=True)
    mul_node = PatternNode("MUL")
    max_node = PatternNode("MAX", const_value=[(-1, [0.0])])
    min_node = PatternNode("MIN", const_value=[(-1, [0.0])])
    add_node.inp_oprs = [max_node, mul_node]
    mul_node.inp_oprs = [min_node]

    add_opr = opr.out_oprs[0]
    if match(add_node, add_opr):
        if (
            max_node.inp_vars[0] == min_node.inp_vars[0]
            and len(mul_node.inp_const) == 1
            and mul_node.inp_const[0][1].shape == (1,)
        ):
            leaky_relu = Ops.LeakyReluOpr(
                "leaky_relu_" + add_node.op.name,
                add_node.op._opr,
                mul_node.inp_const[0][1],
            )
            leaky_relu.inp_vars = [max_node.inp_vars[0][1]]
            leaky_relu.out_vars = add_node.op.out_vars
            leaky_relu.inp_oprs = max_node.op.inp_oprs
            leaky_relu.out_oprs = add_node.op.out_oprs
            for node in [add_node, mul_node, max_node, min_node]:
                node.op.skip = True
            return leaky_relu

    return None


@_register_tranformation_rule(TransformerRule.FUSE_FOR_LEAKY_RELU)
def _(net):
    """
    Elemwise(ADD) + Elemwise(MUL) + Elemwise(MAX) + Elemwise(MIN) -> LeakyRelu
    """
    matches = list()
    for opr in net.all_oprs:
        if (
            get_type(opr) == "MAX"
            and len(opr.out_oprs) == 1
            and get_type(opr.out_oprs[0]) == "ADD"
        ):
            leaky_relu = _fuse_for_leaky_relu(opr)
            if leaky_relu:
                matches.append(leaky_relu)
    _replace_opr(net, matches)


def _fuse_for_conv_bias(opr):
    assert (
        len(opr.out_oprs) == 1
        and isinstance(opr.out_oprs[0], Ops.ElemwiseOpr)
        and opr.out_oprs[0].mode == "ADD"
    )

    bias_node = PatternNode("ADD", is_output=True)
    conv_node = PatternNode(Ops.ConvolutionForwardOpr.__name__)
    bias_node.inp_oprs = [conv_node]

    add_opr = opr.out_oprs[0]
    if match(bias_node, add_opr):
        conv_bias = Ops.ConvForwardBiasOpr(
            "ConvForwardBias_" + bias_node.op.name,
            conv_node.op._opr,
            bias_node.inp_const[0][1],
        )
        conv_bias.activation = add_opr.activation
        conv_bias.inp_vars = conv_node.op.inp_vars + bias_node.op.inp_vars[1:]
        conv_bias.out_vars = bias_node.op.out_vars
        conv_bias.inp_oprs = conv_node.op.inp_oprs
        conv_bias.out_oprs = bias_node.op.out_oprs
        for node in [conv_node, bias_node]:
            node.op.skip = True
        return conv_bias
    return None


@_register_tranformation_rule(TransformerRule.FUSE_FOR_CONV_BIAS)
def _(net):
    """
    ConvolutionForward + Elemwise(ADD) -> ConvForwardBias
    """
    matches = list()
    for opr in net.all_oprs:
        if (
            get_type(opr) == Ops.ConvolutionForwardOpr.__name__
            and len(opr.out_oprs) == 1
            and get_type(opr.out_oprs[0]) == "ADD"
        ):
            conv_bias = _fuse_for_conv_bias(opr)
            if conv_bias:
                matches.append(conv_bias)
    _replace_opr(net, matches)


@_register_tranformation_rule(TransformerRule.RESHAPE_BIAS_TO_1DIM)
def _(net):
    for opr in net.all_oprs:
        if isinstance(opr, Ops.ConvForwardBiasOpr) and opr.inp_vars[2].ndim != 1:
            bias = opr.inp_vars[2]
            assert bias.shape == (1, bias.shape[1], 1, 1), (
                "bias.shape = %s" % bias.shape
            )
            bias.np_data = bias.np_data.reshape(-1)
            bias.shape = bias.np_data.shape
            bias.ndim = 1


def _fuse_for_deconv_bias(opr):
    assert (
        len(opr.out_oprs) == 1
        and isinstance(opr.out_oprs[0], Ops.ElemwiseOpr)
        and opr.out_oprs[0].mode == "ADD"
    )

    bias_node = PatternNode("ADD", is_output=True)
    conv_node = PatternNode(Ops.ConvolutionBackwardDataOpr.__name__)
    bias_node.inp_oprs = [conv_node]

    add_opr = opr.out_oprs[0]
    if match(bias_node, add_opr):
        deconv_bias = Ops.ConvolutionBackwardDataBiasOpr(
            "ConvolutionBackwardDataBias_" + bias_node.op.name,
            conv_node.op._opr,
            bias_node.inp_const[0][1],
        )
        deconv_bias.activation = add_opr.activation
        deconv_bias.inp_vars = conv_node.op.inp_vars + bias_node.op.inp_vars[1:]
        deconv_bias.out_vars = bias_node.op.out_vars
        deconv_bias.inp_oprs = conv_node.op.inp_oprs
        deconv_bias.out_oprs = bias_node.op.out_oprs
        for node in [conv_node, bias_node]:
            node.op.skip = True
        return deconv_bias

    return None


@_register_tranformation_rule(TransformerRule.FUSE_FOR_DECONV_BIAS)
def _(net):
    """
    ConvolutionBackwardData + Elemwise(ADD) -> ConvolutionBackwardDataBias
    """
    matches = list()
    for opr in net.all_oprs:
        if (
            get_type(opr) == Ops.ConvolutionBackwardDataOpr.__name__
            and len(opr.out_oprs) == 1
            and get_type(opr.out_oprs[0]) == "ADD"
        ):
            deconv_bias = _fuse_for_deconv_bias(opr)
            if deconv_bias:
                matches.append(deconv_bias)
    _replace_opr(net, matches)


def _fuse_for_fully_connected(opr):
    assert (
        len(opr.out_oprs) == 1
        and isinstance(opr.out_oprs[0], Ops.ElemwiseOpr)
        and opr.out_oprs[0].mode == "ADD"
    )
    bias_node = PatternNode("ADD", is_output=True)
    matrix_mul_node = PatternNode(Ops.MatrixMulOpr.__name__)
    bias_node.inp_oprs = [matrix_mul_node]

    add_opr = opr.out_oprs[0]
    if match(bias_node, add_opr):
        fully_connected = Ops.FullyConnectedOpr(
            "FullyConnected_" + bias_node.op.name,
            matrix_mul_node.op._opr,
            bias_node.inp_const[0][1],
        )
        fully_connected.activation = add_opr.activation
        fully_connected.inp_vars = (
            matrix_mul_node.op.inp_vars + bias_node.op.inp_vars[1:]
        )
        fully_connected.out_vars = bias_node.op.out_vars
        fully_connected.inp_oprs = matrix_mul_node.op.inp_oprs
        fully_connected.out_oprs = bias_node.op.out_oprs
        for node in [matrix_mul_node, bias_node]:
            node.op.skip = True
        return fully_connected
    return None


@_register_tranformation_rule(TransformerRule.FUSE_FOR_FULLY_CONNECTED)
def _(net):
    """
    MatrixMul + Elemwise(ADD) -> FullyConnected
    """
    matches = list()
    for opr in net.all_oprs:
        if (
            get_type(opr) == Ops.MatrixMulOpr.__name__
            and len(opr.out_oprs) == 1
            and get_type(opr.out_oprs[0]) == "ADD"
        ):
            fc = _fuse_for_fully_connected(opr)
            if fc:
                matches.append(fc)
    _replace_opr(net, matches)


def _replace_opr(net, matches: List[Ops.MgeOpr]):
    """
    Recieve a list of :class:`~.Ops.MgeOpr`.
    For each operator in :attr:`matches`, this function will insert it and its id.

    At the end, delete the orignal operators who matchs the transform rule.
    """
    for opr in matches:
        max_idx = max(net._opr_ids.index(i.id) for i in opr.inp_oprs)
        net._opr_ids.insert(max_idx + 1, opr.id)
        net.all_oprs.insert(max_idx + 1, opr)
    new_idxs = []
    new_oprs = []
    for idx, opr in zip(net._opr_ids, net.all_oprs):
        if opr.skip:
            continue
        new_idxs.append(idx)
        new_oprs.append(opr)
    net._opr_ids = new_idxs
    net.all_oprs = new_oprs
