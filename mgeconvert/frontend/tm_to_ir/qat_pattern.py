# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module

import megengine.functional as F
from megengine.core.ops import builtin
from megengine.module.qat import QATModule
from megengine.traced_module.expr import Apply, CallFunction, GetAttr

from ...converter_ir.ir_op import ReluOpr
from .op_generators import GenConv2dOpr, GenDeconv2dOpr
from .pattern_utils import InputNode, MatchAnyNode, is_match, register_fusion_pattern
from .tm_tensor_resolver import TensorNodeResolver


def gen_qat_conv_opr(module, conv_function_expr, qat_expr, irgraph, is_deconv=False):
    weight_fakequant = conv_function_expr.inputs[1].expr
    bias = None
    if len(conv_function_expr.inputs) == 3:
        bias = conv_function_expr.inputs[2]
        assert (isinstance(bias.expr, GetAttr) and bias.expr.name == "bias") or (
            isinstance(bias.expr, Apply)
            and isinstance(
                bias.expr.opdef, builtin.FakeQuant  # pylint:disable=no-member
            )
        )
    assert (
        isinstance(weight_fakequant.inputs[1].expr, GetAttr)
        and weight_fakequant.inputs[1].expr.name == "weight"
    )
    assert len(module.graph.inputs) == 2

    act_qparams = (
        module.act_fake_quant.get_qparams()
        if module.act_observer is None
        else module.act_observer.get_qparams()
    )
    weight_qparams = (
        module.weight_fake_quant.get_qparams()
        if module.weight_observer is None
        else module.weight_observer.get_qparams()
    )

    module.stride = conv_function_expr.args[3]
    module.padding = conv_function_expr.args[4]
    module.dilation = conv_function_expr.args[5]
    module.groups = conv_function_expr.args[6]
    assert conv_function_expr.args[7] == "cross_correlation"
    assert conv_function_expr.args[8] == "default"
    if bias is None:
        module.bias = None

    op = (
        GenConv2dOpr(qat_expr, irgraph).get_opr()
        if not is_deconv
        else GenDeconv2dOpr(qat_expr, irgraph).get_opr()
    )

    op.inp_tensors[1].set_qparams_from_mge_qparams(weight_qparams)
    if len(op.inp_tensors) == 3:
        op.inp_tensors[2].set_qparams(
            scale=op.inp_tensors[0].scale * op.inp_tensors[1].scale,
            zero_point=0,
            q_dtype="int32",
            np_dtype="int32",
        )
    op.out_tensors[0].set_qparams_from_mge_qparams(act_qparams)
    return op


MATCH_RULE = {}

pat_conv_bias_relu = (
    QATModule._apply_fakequant_with_observer,
    MatchAnyNode,
    (
        F.relu,
        (F.conv2d, InputNode, QATModule._apply_fakequant_with_observer, MatchAnyNode),
    ),
    MatchAnyNode,
)

pat_conv_bias_relu_1 = (
    QATModule._apply_fakequant_with_observer,
    MatchAnyNode,
    (
        F.relu,
        (F.conv2d, InputNode, QATModule._apply_fakequant_with_observer, MatchAnyNode),
    ),
    MatchAnyNode,
    MatchAnyNode,
)

pat_conv_bias = (
    QATModule._apply_fakequant_with_observer,
    MatchAnyNode,
    (F.conv2d, InputNode, QATModule._apply_fakequant_with_observer, MatchAnyNode),
    MatchAnyNode,
)

pat_conv_relu = (
    QATModule._apply_fakequant_with_observer,
    MatchAnyNode,
    (F.relu, (F.conv2d, InputNode, QATModule._apply_fakequant_with_observer),),
    MatchAnyNode,
)

pat_conv = (
    QATModule._apply_fakequant_with_observer,
    MatchAnyNode,
    (F.conv2d, InputNode, QATModule._apply_fakequant_with_observer),
    MatchAnyNode,
)

pat_deconv_relu = (
    QATModule._apply_fakequant_with_observer,
    MatchAnyNode,
    (F.relu, (F.conv_transpose2d, InputNode, QATModule._apply_fakequant_with_observer)),
    MatchAnyNode,
)

pat_deconv_bias = (
    QATModule._apply_fakequant_with_observer,
    MatchAnyNode,
    (
        F.conv_transpose2d,
        InputNode,
        QATModule._apply_fakequant_with_observer,
        MatchAnyNode,
    ),
    MatchAnyNode,
)


@register_fusion_pattern([pat_conv_bias_relu, pat_conv_bias_relu_1])
def qat_conv_bias_relu(module, expr, call_expr, irgraph, _):
    relu = expr.inputs[1].expr
    op = gen_qat_conv_opr(module, relu.inputs[0].expr, call_expr, irgraph)
    op.activation = "RELU"
    return op


@register_fusion_pattern(pat_conv_bias)
def qat_conv_bias(module, expr, call_expr, irgraph, _):
    conv = expr.inputs[1].expr
    op = gen_qat_conv_opr(module, conv, call_expr, irgraph)
    return op


@register_fusion_pattern(pat_conv_relu)
def qat_conv_relu(module, expr, call_expr, net, _):
    relu = expr.inputs[1].expr
    op = gen_qat_conv_opr(module, relu.inputs[0].expr, call_expr, net)
    op.activation = "RELU"
    return op


@register_fusion_pattern(pat_conv)
def qat_conv(module, expr, call_expr, net, _):
    conv = expr.inputs[1].expr
    op = gen_qat_conv_opr(module, conv, call_expr, net)
    return op


@register_fusion_pattern(pat_deconv_bias)
def qat_deconv_bias(module, expr, call_expr, irgraph, _):
    conv = expr.inputs[1].expr
    op = gen_qat_conv_opr(module, conv, call_expr, irgraph, is_deconv=True)
    return op


@register_fusion_pattern(pat_deconv_relu)
def qat_deconv_relu_bias(
    module, expr, call_expr, irgraph, resolver: TensorNodeResolver
):
    relu = expr.inputs[1].expr
    deconv = relu.inputs[0].expr
    op = gen_qat_conv_opr(module, deconv, call_expr, irgraph, is_deconv=True)
    op.activation = "RELU"

    relu_op = ReluOpr()
    relu_op.inp_tensors = []
    relu_op.out_tensors = []
    relu_op.inp_tensors.append(op.out_tensors[0])
    relu_op.out_tensors.append(resolver.resolve(call_expr.outputs[0], relu_op)[0])
    relu_op.out_tensors[0].name += "_relu"

    relu_op.out_tensors[0].set_qparams_from_other_tensor(relu_op.inp_tensors[0])
    irgraph.all_tensors[
        irgraph._tensor_ids.index(call_expr.outputs[0]._id)
    ] = relu_op.out_tensors[0]

    return op, relu_op


MATCH_RULE[QATModule._apply_fakequant_with_observer] = [
    pat_conv_bias_relu,
    pat_conv_bias_relu_1,
    pat_conv_bias,
    pat_deconv_relu,
    pat_conv_relu,
    pat_conv,
    pat_deconv_bias,
]


def find_match_pattern(graph):
    rst = []
    for expr in graph._exprs:
        if isinstance(expr, CallFunction):
            if expr.func in MATCH_RULE:
                pat = MATCH_RULE[expr.func]
                for p in pat:
                    if is_match(expr, p):
                        rst.append((p, expr))
    return rst
