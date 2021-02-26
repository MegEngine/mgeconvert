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
