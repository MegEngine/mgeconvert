# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint:disable=no-name-in-module,import-error
from typing import List, Sequence

import megengine
from megengine.core._imperative_rt.core2 import Tensor as RawTensor
from megengine.module.qat import QATModule, QuantStub
from megengine.module.quant_dequant import DequantStub as FloatDequantStub
from megengine.module.quant_dequant import QuantStub as FloatQuantStub
from megengine.traced_module import TracedModule
from megengine.traced_module.expr import (
    Apply,
    CallFunction,
    CallMethod,
    Constant,
    GetAttr,
    Input,
)
from megengine.traced_module.module_tracer import BUILTIN_ARRAY_METHOD
from megengine.traced_module.node import ModuleNode, TensorNode

from ...converter_ir.ir_graph import IRGraph
from .op_generators import EXPR2OP
from .pattern_utils import DEFAULT_FUSION_PATTERNS
from .qat_pattern import find_match_pattern
from .tm_tensor_resolver import TensorNodeResolver
from .tm_utils import get_logger

logger = get_logger(__name__)


class TM_FrontEnd:
    def __init__(self, traced_module, outspec=None):
        if isinstance(traced_module, TracedModule):
            self.module = traced_module.flatten()
        elif isinstance(traced_module, str):
            self.module = megengine.load(traced_module)

        if outspec is not None:
            target_nodes = set()
            g = self.module.graph
            for name in outspec:
                nodes = g.get_node_by_name(name).as_list()
                if len(nodes) == 0:
                    print("{:p}".format(g))  # type: ignore
                    raise ValueError(
                        "Only allow reset outputs in top graph , cannot find node by the full-name({})".format(
                            name
                        )
                    )
                target_nodes = target_nodes.union(set(nodes))
            g.reset_outputs(list(target_nodes))
            g.compile()

        self.inputs: List[TensorNode] = self.module.graph.inputs[1:]
        self.outputs: List[TensorNode] = self.module.graph.outputs

        self.irgraph = IRGraph()
        self.tensor_resolver = TensorNodeResolver(self.irgraph)
        self.has_qat = False

    def resolve(self):
        self.add_net_inputs()
        self.get_all_oprs()
        self.add_net_outputs()
        return self.irgraph

    def add_net_inputs(self):
        for node in self.inputs:
            inp_tensor = self.tensor_resolver.get_ir_tensor(node, owner_opr=self)
            if node.qparams is not None:
                inp_tensor.set_qparams_from_mge_qparams(node.qparams)

            self.irgraph.add_net_inputs(inp_tensor)

    def get_all_oprs(self):
        for expr in self.module.graph._exprs:
            if isinstance(expr, Constant):
                if isinstance(expr.value, RawTensor):
                    op_gen_cls = EXPR2OP.get("Constant")
                    op = op_gen_cls(expr, self.irgraph).get_opr()
                    self.irgraph.add_op(op)
            elif isinstance(expr, GetAttr):
                if isinstance(expr.outputs[0], TensorNode):
                    op_gen_cls = EXPR2OP.get("Constant")
                    op = op_gen_cls(expr, self.irgraph).get_opr()
                    self.irgraph.add_op(op)
            elif isinstance(expr, CallMethod):
                if expr.method in BUILTIN_ARRAY_METHOD:
                    # generate array_method op
                    op_gen_cls = EXPR2OP.get(expr.method, None)
                    assert op_gen_cls, "METHOD {} is not supported.".format(expr.method)
                    op = op_gen_cls(expr, self.irgraph).get_opr()
                    self.irgraph.add_op(op)
                elif expr.method == "__new__":
                    # TODO
                    pass
                elif expr.method == "__call__":
                    m = expr.inputs[0]
                    assert isinstance(m, ModuleNode)
                    if isinstance(m.owner, TracedModule):
                        self.has_qat = self.has_qat or m.owner.is_qat
                        module = m.owner
                        assert module.is_qat
                        pats = find_match_pattern(module.graph)
                        pat, end_expr = pats[0]
                        fusion_op = DEFAULT_FUSION_PATTERNS.get(pat)
                        ops = fusion_op(
                            module, end_expr, expr, self.irgraph, self.tensor_resolver,
                        )
                        ops = (ops,) if not isinstance(ops, Sequence) else ops
                        for op in ops:
                            self.irgraph.all_oprs.append(op)
                            self.irgraph._opr_ids.append(id(op))
                    elif isinstance(m.owner, QuantStub):
                        module = m.owner
                        inp_tensor = self.tensor_resolver.get_ir_tensor(expr.inputs[1])
                        out_tensor = self.irgraph.get_tensor(
                            expr.outputs[0]._id, None, origin_tensor=inp_tensor
                        )
                        qparams = (
                            module.act_fake_quant.get_qparams()
                            if hasattr(module.act_fake_quant, "get_qparams")
                            else module.act_observer.get_qparams()
                        )
                        out_tensor.set_qparams_from_mge_qparams(qparams)
                    elif isinstance(m.owner, (FloatQuantStub, FloatDequantStub)):
                        module = m.owner
                        inp_tensor = self.tensor_resolver.get_ir_tensor(expr.inputs[1])
                        self.irgraph.get_tensor(
                            expr.outputs[0]._id, None, origin_tensor=inp_tensor
                        )
                    else:
                        self.has_qat = self.has_qat or isinstance(m.owner, QATModule)
                        op_gen_cls = EXPR2OP.get(type(m.owner), None)
                        assert op_gen_cls, "Module {} is not supported.".format(
                            type(m.owner)
                        )
                        op = op_gen_cls(expr, self.irgraph).get_opr()
                        self.irgraph.add_op(op)
            elif isinstance(expr, CallFunction):
                f = expr.func  # expr.func.__module__ + "." + expr.func.__name__
                op_gen_cls = EXPR2OP.get(f, None)
                assert op_gen_cls, "FUNCTION {} is not supported.".format(f)
                op = op_gen_cls(expr, self.irgraph).get_opr()
                self.irgraph.add_op(op)
            elif isinstance(expr, Apply):
                opdef = expr.opdef
                op_gen_cls = EXPR2OP.get(str(opdef), None)
                assert op_gen_cls, "OPDEF {} is not supported.".format(str(opdef))
                op = op_gen_cls(expr, self.irgraph).get_opr()
                self.irgraph.add_op(op)
            elif isinstance(expr, Input):
                logger.warning("Do not suppot Input Expr.")

    def add_net_outputs(self):
        for node in self.outputs:
            assert (
                node._id in self.irgraph._tensor_ids
            ), "output node is not generated by any opr"
            out_tensor = self.tensor_resolver.get_ir_tensor(node, self)
            self.irgraph.add_net_outputs(out_tensor)
