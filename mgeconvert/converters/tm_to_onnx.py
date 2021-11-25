# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module,no-member

from typing import List, Union

import megengine as mge
from megengine.traced_module import TracedModule

from ..backend.ir_to_onnx.onnx_converter import OnnxConverter
from ..converter_ir.ir_quantizer import IRQuantizer
from ..converter_ir.ir_transform import IRTransform, TransformerRule
from ..frontend.tm_to_ir import TM_FrontEnd


def tracedmodule_to_onnx(
    traced_module,
    output="out.onnx",
    *,
    graph_name="graph",
    opset=8,
    outspec=None,
    input_data_type: str = None,
    input_scales: Union[float, List[float]] = None,
    input_zero_points: Union[int, List[int]] = None,
    require_quantize=False,
    param_fake_quant=False,
    quantize_file_path="quant_params.json",
):
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
    if isinstance(traced_module, str):
        traced_module = mge.load(traced_module)
    assert isinstance(
        traced_module, TracedModule
    ), "Input should be a traced module or a path of traced module."

    irgraph = TM_FrontEnd(traced_module, outspec=outspec).resolve()
    irgraph.update_inputs_qparams(input_data_type, input_scales, input_zero_points)

    transformer_options = [
        TransformerRule.REMOVE_RESHAPE_REALTED_OP,
        TransformerRule.REMOVE_UNRELATED_IROP,
        TransformerRule.EXPAND_CONVRELU,
    ]
    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)

    if require_quantize:
        quantizer = IRQuantizer(
            require_quantize=require_quantize, param_fake_quant=param_fake_quant
        )
        quantizer.save_quantize_params(transformed_irgraph)
        converter = OnnxConverter(transformed_irgraph, opset, graph_name, quantizer)
    else:
        converter = OnnxConverter(transformed_irgraph, opset, graph_name)

    model = converter.convert()

    if require_quantize:
        quantizer.dump_quant_param(path=quantize_file_path)

    assert isinstance(output, str), "onnx_fpath must be string"
    with open(output, "wb") as fout:
        fout.write(model.SerializeToString())
