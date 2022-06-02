# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Union

import megengine as mge
import onnx.checker
import onnx.helper
import onnx.numpy_helper
import onnxoptimizer  # pylint: disable=import-error,no-name-in-module
from tqdm import tqdm

from ...converter_ir.ir_quantizer import IRQuantizer
from ...frontend.mge_to_ir.mge_utils import get_symvar_value
from .onnx_op import (
    MGE2ONNX,
    _add_input_tensors,
    mge2onnx_dtype_mapping,
    set_opset_version,
)


class OnnxConverter:
    def __init__(
        self,
        net,
        opset_version=8,
        graph_name="graph",
        quantizer: Union[IRQuantizer, None] = None,
    ):
        self.net = net
        assert 7 <= opset_version <= 12, "opset {} are not supported yet".format(
            opset_version
        )
        self.graph_name = graph_name
        self.opset_version = opset_version
        self.quantizer = quantizer

    def convert(self):
        inputs = []
        parameters = []
        onnx_nodes = []
        outputs = []
        unsupported_oprs = []
        set_opset_version(self.opset_version)

        def need_convert(opr):
            is_const = [data.np_data is not None for data in opr.inp_tensors]
            return not all(is_const) or len(opr.inp_tensors) == 0

        def deduplication(inputs):
            names = []
            results = []
            for i in inputs:
                if i.name not in names:
                    results.append(i)
                    names.append(i.name)
            return results

        _, tensor_sources, _ = _add_input_tensors(self.net.graph_inputs)
        inputs.extend(tensor_sources)

        for opr in tqdm(self.net.all_oprs):
            if not need_convert(opr):
                for tensor in opr.out_tensors:
                    if hasattr(tensor, "_var"):
                        tensor.np_data = get_symvar_value(tensor._var)
                continue
            converter_cls = MGE2ONNX.get(type(opr), None)
            if converter_cls is None:
                unsupported_oprs.append(opr)
                continue
            converter = converter_cls(opr, self.quantizer)
            nodes, inps, params = converter.convert()
            onnx_nodes.extend(nodes)
            inputs.extend(inps)
            parameters.extend(params)

        inputs = deduplication(inputs)
        parameters = deduplication(parameters)

        unsupported_oprs = set(map(type, unsupported_oprs))
        assert not unsupported_oprs, "Operators {} are not supported yet".format(
            unsupported_oprs
        )

        for output in self.net.graph_outputs:

            def _get_onnx_dtype(output):
                return mge2onnx_dtype_mapping[output.dtype]

            out_tensor = onnx.helper.make_tensor_value_info(
                output.name, _get_onnx_dtype(output), output.shape
            )
            outputs.append(out_tensor)

        onnx_graph = onnx.helper.make_graph(
            onnx_nodes, self.graph_name, inputs, outputs, initializer=parameters
        )
        opset = onnx.helper.make_opsetid("", self.opset_version)
        model = onnx.helper.make_model(
            onnx_graph,
            producer_name="MegEngine",
            producer_version=mge.__version__,
            opset_imports=[opset],
        )
        onnx.checker.check_model(model)
        passes = [
            "eliminate_deadend",
            "extract_constant_to_initializer",
            "eliminate_unused_initializer",
        ]
        model = onnxoptimizer.optimize(model, passes=passes)
        return model
