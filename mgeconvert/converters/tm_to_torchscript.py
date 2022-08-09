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

from ..backend.ir_to_torchscript import PytorchConverter
from ..converter_ir.ir_quantizer import IRQuantizer
from ..converter_ir.ir_transform import IRTransform, TransformerRule
from ..frontend.tm_to_ir import TM_FrontEnd
from ..frontend.tm_to_ir.tm_utils import _update_inputs_qparams


def tracedmodule_to_torchscript(
    traced_module,
    output="out.torchscript",
    *,
    input_data_type: str = None,
    input_scales: Union[float, List[float]] = None,
    input_zero_points: Union[int, List[int]] = None,
    graph_name="graph",
    outspec=None,
):
    """
	Convert traced model to torchscript,
	and save the torchscript model to file `output`.

	:param traced_module: a traced module or the file path of a traced module.
	:param output: the filename used for the saved model.
	:param input_data_type: data type of input
	:param graph_name: the name of the torchscript graph.
	"""
    if isinstance(traced_module, str):
        traced_module = mge.load(traced_module)
    assert isinstance(
        traced_module, TracedModule
    ), "Input should be a traced module or a path of traced module."
    _update_inputs_qparams(
        traced_module, input_data_type, input_scales, input_zero_points
    )
    tm_resolver = TM_FrontEnd(traced_module, outspec=outspec)
    irgraph = tm_resolver.resolve()

    transformer_options = [
        TransformerRule.REMOVE_IDENTITY,
        TransformerRule.REPLACE_FLATTEN_TO_RESHAPE,
        TransformerRule.RESHAPE_BIAS_TO_1DIM,
        TransformerRule.EXPAND_ADD_RELU,
    ]

    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)

    quantizer = IRQuantizer(require_quantize=True, param_fake_quant=False)
    converter = PytorchConverter(transformed_irgraph, graph_name, quantizer=quantizer)
    model = converter.convert()
    model.save(output)
