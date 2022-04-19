# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module

from collections import OrderedDict
from functools import partial
from typing import Union

from megengine.logger import get_logger
from megengine.module import Module
from megengine.traced_module.expr import (
    Apply,
    CallFunction,
    CallMethod,
    Constant,
    Expr,
    Input,
)
from megengine.traced_module.node import ModuleNode, Node
from megengine.traced_module.pytree import SUPPORTED_TYPE, LeafDef, tree_flatten

logger = get_logger(__name__)

DEFAULT_FUSION_PATTERNS = OrderedDict()  # type: OrderedDict


def flatten(treedef, values):
    if isinstance(treedef, LeafDef):
        return [
            values,
        ]
    if type(values) != treedef.type:  # pylint:disable=unidiomatic-typecheck
        return None
    rst = []
    children_values, aux_data = SUPPORTED_TYPE[treedef.type].flatten(values)
    if treedef.aux_data != aux_data:
        return None
    for ch_def, ch_val in zip(treedef.children_defs, children_values):
        v_list = flatten(ch_def, ch_val)
        if v_list is None:
            return None
        rst.extend(v_list)
    if treedef.num_leaves != len(rst):
        return None
    return rst


class MatchAnyNode:
    pass


class InputNode:
    pass


def register_pattern(pattern, default_dict: OrderedDict):
    def insert(func):
        if isinstance(pattern, list):
            for p in pattern:
                default_dict[p] = func
        else:
            default_dict[pattern] = func
        return func

    return insert


register_fusion_pattern = partial(
    register_pattern, default_dict=DEFAULT_FUSION_PATTERNS
)


def check_match(expr: Expr, target):  # pylint:disable=too-many-return-statements
    if isinstance(target, type) and issubclass(target, MatchAnyNode):
        return True

    if target is InputNode and isinstance(expr, Input):
        return True

    if isinstance(expr, Apply):
        opdef = expr.opdef
        return isinstance(opdef, target)

    if isinstance(target, type) and issubclass(target, Module):
        if not isinstance(expr, (CallMethod, Constant)):
            return False
        if isinstance(expr, CallMethod):
            obj_node = expr.inputs[0]
            if not isinstance(obj_node, ModuleNode):
                return False
            obj = obj_node.owner
        else:
            obj = expr.value
        if type(obj) != target:  # pylint:disable=unidiomatic-typecheck
            return False
    elif callable(target):
        if not isinstance(expr, CallFunction):
            return False
        if expr.func != target:
            return False
    elif isinstance(target, str):
        if not isinstance(expr, CallMethod):
            return False
        if expr.method != target:
            return False
    else:
        if expr != target:
            return False

    return True


def is_match(
    expr: Union[Expr, Node], pattern: tuple, max_uses=100
):  # pylint:disable=too-many-return-statements

    if isinstance(pattern, tuple):
        self_match, *arg_matches = pattern
    else:
        self_match = pattern
        arg_matches = []

    if isinstance(expr, Node):
        expr = expr.expr

    if isinstance(expr, Expr):
        if max_uses == 1:
            if len(expr.outputs) > 1:
                logger.warning("is_match only support 1 output expr")
                return False
        for n in expr.outputs:
            if len(n.users) > max_uses:
                return False

    if not check_match(expr, self_match):
        return False

    if not arg_matches:
        return True

    expr_args = expr.inputs

    if isinstance(expr, CallMethod) and isinstance(expr_args[0], ModuleNode):
        expr_args = expr_args[1:]
    if len(expr_args) != len(arg_matches):
        return False

    for inp_node, arg_match in zip(expr_args, arg_matches):
        inp_node, inp_def = tree_flatten(inp_node)
        inp_match = flatten(inp_def, arg_match)
        if inp_match is None:
            return False
        for node, arg in zip(inp_node, inp_match):
            if isinstance(node, ModuleNode):
                continue
            if not is_match(node, arg, max_uses=1):
                return False

    return True
