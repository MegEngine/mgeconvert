# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import numpy as np
import onnx
import onnxruntime
import onnxsim

from ...converter_ir.ir_graph import IRGraph
from .onnxproto_resolver import ONNXProtoResolver
from .op_generators import ONNX2OP

ONNXDTYPE2NUMPY = {
    "tensor(float)": np.float32,
    "tensor(uint16)": np.uint16,
    "tensor(int16)": np.int16,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
}


class ONNX_FrontEnd:
    def __init__(self, onnx_model_path):
        self.onnx_model, _ = onnxsim.simplify(onnx_model_path)
        self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
        print(f"ONNX Model Producer : {self.onnx_model.producer_name}")
        print(f"ONNX Model Producer Version: {self.onnx_model.producer_version}")
        print(f"ONNX Model IR Version : {self.onnx_model.ir_version}")
        if self.onnx_model.model_version:
            print(f"ONNX Model Version : {self.onnx_model.model_version}")
        if self.onnx_model.doc_string:
            print(self.onnx_model.doc_string)

        self.onnx_opset = self.onnx_model.opset_import[0].version
        print(f"ONNX Model OpSet : {self.onnx_opset}")

        self.ir_graph = IRGraph()
        self.resolver = ONNXProtoResolver(self.onnx_model)
        self.onnx_session = onnxruntime.InferenceSession(onnx_model_path)

    def is_model_input(self, node):
        for tensor in self.onnx_model.graph.initializer:
            if node.name == tensor.name:
                return False
        return True

    def resolve(self):
        for input in self.onnx_model.graph.input:
            if self.is_model_input(input):
                ir_tensor = self.resolver.get_ir_tensor_from_valueinfo(input)
                # find duplicated ir tensor or add to ir graph otherwise
                ori_ir_tensor = self.ir_graph.get_tensor(
                    hash(ir_tensor.name), ir_tensor
                )
                self.ir_graph.add_net_inputs(ori_ir_tensor)

        # one node represent one operator
        for node in self.onnx_model.graph.node:
            self.add_node(node)

        for output in self.onnx_model.graph.output:
            ir_tensor = self.resolver.get_ir_tensor_from_valueinfo(output)
            # find duplicated ir tensor or add to ir graph otherwise
            ori_ir_tensor = self.ir_graph.get_tensor(hash(ir_tensor.name), ir_tensor)
            self.ir_graph.add_net_outputs(ori_ir_tensor)

        return self.ir_graph

    def add_node(self, node):
        op_gen_cls = ONNX2OP.get(node.op_type, None)
        assert op_gen_cls, f"OP {node.op_type} is not supported"
        ir_opr = op_gen_cls(
            node, self.ir_graph, self.resolver, self.onnx_opset
        ).get_opr()

        self.ir_graph.add_op(ir_opr)

    def eval(self, seed: int):
        np.random.seed(seed)
        inputs = []
        inputs_feed = {}
        for node in self.onnx_session.get_inputs():
            tensor = np.random.randint(
                low=0, high=100, size=node.shape, dtype="int32"
            ).astype(ONNXDTYPE2NUMPY[node.type])
            inputs_feed[node.name] = tensor
            inputs.append(node.name)

        outputs = [node.name for node in self.onnx_session.get_outputs()]
        print(f"onnx input(s) name : {inputs}")
        print(f"onnx output(s) name : {outputs}")
        return self.onnx_session.run(outputs, inputs_feed)
