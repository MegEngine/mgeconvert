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

import numpy as np

from .mge_net import TopologyNetwork
from .mge_tensor import FakeSymbolVar


class TransformerRule(Enum):
    # general rules
    NOPE = 1

    # for TFLite
    REDUCE_AXIS_AS_INPUT = 100
    RESIZE_SHAPE_AS_INPUT = 101
    # FUSE_FOR_RELU6 pass should happen before FUSE_ACTIVATION
    FUSE_FOR_RELU6 = 102
    FUSE_ACTIVATION = 103
    CONV_ADD_ZERO_BIAS = 104
    DEPTHWISE_CONV_RESHAPE_WEIGHT = 105
    FUSE_SOFTMAX = 106
    DECONV_SHAPE_AS_INPUT = 107
    FUSE_ASTYPE = 108

    # for Caffe
    FUSE_FOR_LEAKY_RELU = 200


TRANSFORMMAP = {}


def optimize_for_conversion(net, transformer_options):
    for option in transformer_options:
        TRANSFORMMAP[option](net)


def _register_tranformation_rule(transformer_option):
    def callback(impl):
        TRANSFORMMAP[transformer_option] = impl

    return callback


@_register_tranformation_rule(TransformerRule.REDUCE_AXIS_AS_INPUT)
def _reduce_axis_as_input(net):
    from .mge_op import ReduceOpr

    for op_id, op in zip(net._opr_ids, net.all_oprs):
        if type(op) != ReduceOpr:
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


@_register_tranformation_rule(TransformerRule.RESIZE_SHAPE_AS_INPUT)
def _resize_shape_as_input(net):
    pass


@_register_tranformation_rule(TransformerRule.FUSE_FOR_RELU6)
def _fuse_for_relu6(net):
    from .mge_op import ElemwiseOpr, Relu6Opr

    matches = OrderedDict()

    for op_id, op in zip(net._opr_ids, net.all_oprs):
        if type(op) != ElemwiseOpr:
            continue
        if len(op.inp_oprs) <= 0 or len(op.inp_oprs) >= 2:
            continue
        prev_op = op.prev_opr

        if op.mode == "MIN" and np.array_equal(op.inp_vars[1].np_data, np.array([6])):
            if (
                type(prev_op) == ElemwiseOpr
                and prev_op.mode == "MAX"
                and np.array_equal(prev_op.inp_vars[1].np_data, np.array([0]))
            ):
                relu6_opr = Relu6Opr()
                relu6_opr.inp_vars = prev_op.inp_vars
                relu6_opr.out_vars = op.out_vars
                relu6_opr.prev_opr = prev_op.prev_opr
                matches[prev_op.id] = (net.max_id, relu6_opr)
                net.max_id += 1
        if op.mode == "MAX" and np.array_equal(op.inp_vars[1].np_data, np.array([0])):
            if (
                type(prev_op) == ElemwiseOpr
                and prev_op.mode == "MIN"
                and np.array_equal(prev_op.inp_vars[1].np_data, np.array([6]))
            ):
                relu6_opr = Relu6Opr()
                relu6_opr.inp_vars = prev_op.inp_vars
                relu6_opr.out_vars = op.out_vars
                relu6_opr.prev_opr = prev_op.prev_opr
                matches[prev_op.id] = (net.max_id, relu6_opr)
                net.max_id += 1

    for original_id, generated_pair in matches.items():
        index = net._opr_ids.index(original_id)
        del net._opr_ids[index : index + 2]
        del net.all_oprs[index : index + 2]

        net._opr_ids.insert(index, generated_pair[0])
        net.all_oprs.insert(index, generated_pair[1])


@_register_tranformation_rule(TransformerRule.FUSE_ACTIVATION)
def _fuse_activation(net):
    from .mge_op import ElemwiseOpr, Relu6Opr

    delete_intended = []

    for op_id, op in zip(net._opr_ids, net.all_oprs):
        if (type(op) == Relu6Opr) or (
            type(op) == ElemwiseOpr and op.mode in ("RELU", "TANH")
        ):
            prev_op = op.prev_opr

            # activation(relu/relu6/tanh) must be fused with previous opr
            activation = getattr(op, "mode", "IDENTITY")
            activation = "RELU6" if (type(op) == Relu6Opr) else activation
            prev_op.activation = activation
            prev_op.out_vars = op.out_vars

            delete_intended.append(net._opr_ids.index(op_id))

    for delete_idx in delete_intended[::-1]:
        del net._opr_ids[delete_idx]
        del net.all_oprs[delete_idx]


@_register_tranformation_rule(TransformerRule.CONV_ADD_ZERO_BIAS)
def _conv_add_zero_bias(net):
    from .mge_op import ConvolutionForwardOpr

    for op in net.all_oprs:
        if type(op) not in (ConvolutionForwardOpr,):
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
    from .mge_op import ConvolutionForwardOpr

    for op in net.all_oprs:
        if type(op) != ConvolutionForwardOpr:
            continue
        if op.group == 1:
            continue

        var = op.inp_vars[1]
        group = var.shape[0]
        oc, ic = var.shape[1:3]
        h, w = var.shape[3:5]
        shape = [1, group * ic, h, w]
        var.ndim = len(shape)
        var.shape = shape
        var.np_data.reshape(shape)


@_register_tranformation_rule(TransformerRule.FUSE_SOFTMAX)
def _fuse_softmax(net):
    from .mge_op import ElemwiseOpr, ReduceOpr, SoftmaxOpr

    matches = OrderedDict()

    for op in net.all_oprs:
        if type(op) != ElemwiseOpr or op.mode != "TRUE_DIV":
            continue
        prev_op = op.inp_oprs[1]
        if type(prev_op) != ReduceOpr or prev_op.mode != "SUM" or prev_op.axis != 1:
            continue
        prev_op = op.inp_oprs[0]
        if type(prev_op) != ElemwiseOpr or prev_op.mode != "EXP":
            continue
        prev_op = prev_op.prev_opr
        if type(prev_op) != ElemwiseOpr or prev_op.mode != "SUB":
            continue
        prev_op = prev_op.inp_oprs[1]
        if type(prev_op) != ReduceOpr or prev_op.mode != "MAX" or prev_op.axis != 1:
            continue

        softmax_opr = SoftmaxOpr()
        softmax_opr.beta = 1
        softmax_opr.inp_vars = prev_op.inp_vars[:1]
        softmax_opr.out_vars = op.out_vars
        softmax_opr.prev_opr = prev_op.inp_oprs[0]
        matches[prev_op.id] = (net.max_id, softmax_opr)
        net.max_id += 1

    for original_id, generated_pair in matches.items():
        index = net._opr_ids.index(original_id)
        del net._opr_ids[index : index + 5]
        del net.all_oprs[index : index + 5]

        net._opr_ids.insert(index, generated_pair[0])
        net.all_oprs.insert(index, generated_pair[1])


@_register_tranformation_rule(TransformerRule.DECONV_SHAPE_AS_INPUT)
def _deconv_shape_as_input(net):
    from .mge_op import ConvolutionBackwardDataOpr

    for op in net.all_oprs:
        if type(op) != ConvolutionBackwardDataOpr:
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


@_register_tranformation_rule(TransformerRule.FUSE_ASTYPE)
def _fuse_astype(net):
    from .mge_op import TypeCvtOpr

    def check_dtype(opr, dtype1, dtype2):
        if opr.inp_vars[0].dtype == dtype1 and opr.out_vars[0].dtype == dtype2:
            return True
        return False

    opr_with_quant = set()

    for op in net.all_oprs:
        if type(op) == TypeCvtOpr:
            prev_op = op.prev_opr
            if (
                check_dtype(op, np.float32, np.int32)
                or check_dtype(op, np.float32, np.uint8)
                or check_dtype(op, np.int32, np.uint8)
                or check_dtype(op, np.uint8, np.uint8)
            ):
                prev_op.out_vars[0] = opr.output_vars[0]
                opr.out_vars[0].owner = prev_op
                opr_with_quant.add(prev_op)
            else:
                assert check_dtype(op, np.uint8, np.int32) or check_dtype(
                    op, np.uint8, np.float32
                ), "ERROR: unsupported Astype mode: {0} to {1}".format(
                    op.inp_vars[0].dtype, opr.out_vars[0].dtype
                )
            continue

        for i in range(len(op.inp_vars)):
            prev_op = op.inp_oprs[i]
            if type(prev_op) == TypeCvtOpr and prev_op.inp_vars[0].dtype == np.uint8:
                # this Astype opr is dequant
                op.inp_vars[i] = prev_op.inp_vars[0]
                result_shape = op.out_vars[0].shape
                out_symvar = FakeSymbolVar(
                    sid=net.max_id,
                    name=op.out_vars[0].name,
                    shape=[
                        result_shape[0],
                        result_shape[2],
                        result_shape[3],
                        result_shape[1],
                    ],
                    dtype=prev_op.inp_vars[0].dtype,
                    owner=op,
                    byte_list=None,
                )
                net.max_id += 1
                out_tensor = net.get_var(out_symvar)
                op.out_vars[0] = out_tensor
