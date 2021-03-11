# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from collections import OrderedDict
from enum import Enum
from typing import Callable, Dict

import numpy as np

from .mge_op import (
    ConvBiasForwardOpr,
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
    MAKE_PADDING = 109
    TRANSPOSE_PATTERN_AS_INPUT = 110
    EXPAND_MUL_ADD3 = 111
    EXPAND_ADD_SIGMOID = 112

    # for Caffe
    FUSE_FOR_LEAKY_RELU = 200


TRANSFORMMAP: Dict[Enum, Callable] = {}


def optimize_for_conversion(net, transformer_options):
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

        if op.mode == "MIN" and np.array_equal(op.inp_vars[1].np_data, np.array([6])):
            if (
                isinstance(prev_op, ElemwiseOpr)
                and prev_op.mode == "MAX"
                and np.array_equal(prev_op.inp_vars[1].np_data, np.array([0]))
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

            # activation(relu/relu6/tanh) must be fused with previous opr
            activation = getattr(op, "mode", "IDENTITY")
            activation = "RELU6" if isinstance(op, Relu6Opr) else activation
            prev_op.activation = activation
            prev_op.out_vars = op.out_vars
            if len(op.out_oprs) > 0:
                idx = op.out_oprs[0].inp_oprs.index(op)
                op.out_oprs[0].inp_oprs[idx] = prev_op
                prev_op.out_oprs = [op.out_oprs[0]]
            else:
                prev_op.out_oprs = []

            delete_intended.append(net._opr_ids.index(op_id))

    for delete_idx in delete_intended[::-1]:
        del net._opr_ids[delete_idx]
        del net.all_oprs[delete_idx]


@_register_tranformation_rule(TransformerRule.CONV_ADD_ZERO_BIAS)
def _conv_add_zero_bias(net):
    for op in net.all_oprs:
        if not isinstance(op, ConvolutionForwardOpr):
            continue
        if isinstance(op, ConvBiasForwardOpr):
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
            if (
                not isinstance(prev_op, ReduceOpr)
                or prev_op.mode != "SUM"
                or prev_op.axis != 1
            ):
                continue
            prev_op = op.inp_oprs[0]
            if not isinstance(prev_op, ElemwiseOpr) or prev_op.mode != "EXP":
                continue
            prev_op = prev_op.prev_opr
            if not isinstance(prev_op, ElemwiseOpr) or prev_op.mode != "SUB":
                continue
            prev_op = prev_op.inp_oprs[1]
            if (
                not isinstance(prev_op, ReduceOpr)
                or prev_op.mode != "MAX"
                or prev_op.axis != 1
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


@_register_tranformation_rule(TransformerRule.MAKE_PADDING)
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
