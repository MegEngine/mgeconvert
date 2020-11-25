# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine as mge
import onnx.checker
import onnx.helper
import onnx.numpy_helper
from onnx import optimizer

from ..mge_context import TopologyNetwork
from .onnx_op import MGE2ONNX, mge2onnx_dtype_mapping, set_opset_version


class OnnxConverter:
    def __init__(self, toponet, opset_version=8, graph_name="graph"):
        assert isinstance(
            toponet, TopologyNetwork
        ), "net must be instance of TopologyNetwork"
        self.net = toponet
        assert (
            opset_version >= 7 and opset_version <= 12
        ), "opset {} are not supported yet".format(opset_version)
        self.graph_name = graph_name
        self.opset_version = opset_version

    def convert(self):
        inputs = []
        parameters = []
        onnx_nodes = []
        outputs = []
        unsupported_oprs = []
        set_opset_version(self.opset_version)

        def need_convert(opr):
            is_const = [data.np_data is not None for data in opr.inp_vars]
            return not all(is_const) or len(opr.inp_vars) == 0

        def deduplication(inputs):
            names = []
            results = []
            for i in inputs:
                if i.name not in names:
                    results.append(i)
                    names.append(i.name)
            return results

        for opr in self.net.all_oprs_map.values():
            if not need_convert(opr):
                continue
            converter_cls = MGE2ONNX.get(type(opr), None)
            if converter_cls is None:
                unsupported_oprs.append(opr)
                continue
            converter = converter_cls(opr)
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

        for output in self.net.output_vars:

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
        model = optimizer.optimize(model, passes)
        return model


def convert_to_onnx(mge_fpath, output="out.onnx", *, graph_name="graph", opset=8):
    """
    Convert megengine model to ONNX,
    and save the ONNX model to file `output`.

    :param mge_fpath: the file path of megengine model.
    :type fpath: str
    :param output: the filename used for the saved model.
    :type output: str
    :param graph_name: the name of the ONNX graph.
    :type graph_name: str
    :param opset: opset version of ONNX model.
    :type opset: int
    """
    assert isinstance(mge_fpath, str), "mge_fpath must be string"
    net = TopologyNetwork(mge_fpath)
    converter = OnnxConverter(net, opset)
    model = converter.convert()

    assert isinstance(output, str), "onnx_fpath must be string"
    with open(output, "wb") as fout:
        fout.write(model.SerializeToString())
