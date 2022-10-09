# pylint: disable=import-error,no-name-in-module,no-member
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.fx
import torch.jit
import tqdm

from ...converter_ir.ir_graph import IRGraph
from ...converter_ir.ir_quantizer import IRQuantizer
from ...converter_ir.ir_tensor import IRTensor
from .fx_op import DTYPE_MAPPING, MGE2FX, make_func_node


def _detect_unsupported_operators(graph):
    r"""Detect all unsupported operators at once"""
    unsupported_ops = set()
    for opr in graph.all_oprs:
        if type(opr) not in MGE2FX:  # pylint:disable=unidiomatic-typecheck
            unsupported_ops.add(type(opr).__name__)

    if unsupported_ops:
        raise RuntimeError(
            "Cannot convert the following operators: {}."
            "".format(", ".join(sorted(unsupported_ops)))
        )


class PytorchConverter:
    def __init__(
        self, net, graph_name="graph", quantizer: Optional[IRQuantizer] = None
    ):
        assert isinstance(net, IRGraph), "net must be instance of IRGraph"
        self.net = net
        self.graph_name = graph_name
        self.module = torch.nn.Module()
        self.graph = torch.fx.Graph()
        self.quantizer = quantizer
        self.env: Dict[str, Any] = {}

    def get_quant_info(self, tensor: IRTensor):
        if self.quantizer is None:
            return None
        if not tensor.q_dtype:
            return None
        if not self.quantizer.require_quantize:
            return None
        return self.quantizer.get_quant_info(tensor)

    def load_arg(
        self, tensor: IRTensor, is_graph_input: bool = False, enable_quant: bool = True
    ):
        name = tensor.name
        node = self.env.get(name, None)

        if node is None:
            if is_graph_input:
                node = self.graph.placeholder(name)
            else:
                assert tensor.np_data is not None
                setattr(self.module, name, torch.tensor(tensor.np_data))
                node = self.graph.get_attr(name)

        q_info, q_name = self.get_quant_info(tensor), name + "_fq"

        if enable_quant and q_info and self.env.get(q_name, None) is None:
            node = self.insert_fakequant(node, name, q_info)

        self.env[name] = node
        return node

    def insert_fakequant(self, node, name, quant_info):
        scale = quant_info["scale"]
        zp = quant_info["zero_point"] if quant_info["zero_point"] else 0
        scale = torch.tensor(np.reshape(np.array(scale), [-1]).astype(np.float32))
        zp = torch.tensor(np.reshape(np.array(zp), [-1]).astype(np.float32))
        scale_name, zp_name = name + "_s", name + "_zp"
        setattr(self.module, scale_name, scale)
        setattr(self.module, zp_name, zp)
        scale_node = self.graph.get_attr(scale_name)
        zp_node = self.graph.get_attr(zp_name)
        fq_node = make_func_node(
            name=name + "_fq",
            graph=self.graph,
            func=torch.fake_quantize_per_tensor_affine,
            input=node,
            quant_min=quant_info["qmin"],
            quant_max=quant_info["qmax"],
            scale=scale_node,
            zero_point=zp_node,
        )
        assert fq_node.name not in self.env
        self.env[fq_node.name] = fq_node
        return fq_node

    def convert(self,):

        _detect_unsupported_operators(self.net)

        for inp in self.net.graph_inputs:
            self.load_arg(inp, is_graph_input=True)

        for opr in tqdm.tqdm(self.net.all_oprs):
            outs = MGE2FX[type(opr)](opr, self.graph, self.load_arg)
            for t, n in zip(opr.out_tensors, outs):
                assert t.name not in self.env
                self.env[t.name] = n

        for out in self.net.graph_outputs:
            out = self.load_arg(out)
            self.graph.output(out)

        module = torch.fx.GraphModule(self.module, self.graph, self.graph_name)
        module.eval()
        inps = [
            torch.zeros(size=n.shape, dtype=DTYPE_MAPPING[n.dtype])
            for n in self.net.graph_inputs
        ]
        script_module = torch.jit.trace(module, tuple(inps))
        return script_module
