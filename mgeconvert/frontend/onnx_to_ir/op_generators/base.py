# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from abc import ABC

from mgeconvert.converter_ir.ir_op import OpBase

ONNX2OP = {}


def _register_op(*ops):
    def callback(impl):
        for op in ops:
            ONNX2OP[op] = impl
        return impl

    return callback


class OpGenBase(ABC):
    def __init__(self, node, ir_graph, resolver) -> None:
        self.node = node
        self.ir_graph = ir_graph
        self.resolver = resolver
        self.op = OpBase()

    def get_opr(self):
        return self.op

    def add_tensors(self):
        # set inp var
        for name in self.node.input:
            ir_tensor = self.resolver.get_ir_tensor_from_name(name)
            ori_ir_tensor = self.ir_graph.get_tensor(hash(ir_tensor.name), ir_tensor)
            if self.op not in ori_ir_tensor.user_opr:
                ori_ir_tensor.add_user_opr(self.op)
            self.op.add_inp_tensors(ori_ir_tensor)

        # set out var
        for name in self.node.output:
            ir_tensor = self.resolver.get_ir_tensor_from_name(name, owner_opr=self.op)
            ori_ir_tensor = self.ir_graph.get_tensor(hash(ir_tensor.name), ir_tensor)
            self.op.add_out_tensors(ori_ir_tensor)
