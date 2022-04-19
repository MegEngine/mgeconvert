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

from ..backend.ir_to_tflite import TFLiteConverter, set_platform
from ..converter_ir.ir_quantizer import IRQuantizer
from ..converter_ir.ir_transform import (
    IRTransform,
    TransformerRule,
    set_conv_pad_perference,
)
from ..frontend.tm_to_ir import TM_FrontEnd
from ..frontend.tm_to_ir.tm_utils import _update_inputs_qparams


def tracedmodule_to_tflite(
    traced_module,
    output="out.tflite",
    *,
    input_data_type: str = None,
    input_scales: Union[float, List[float]] = None,
    input_zero_points: Union[int, List[int]] = None,
    require_quantize=False,
    param_fake_quant=False,
    quantize_file_path="quant_params.json",
    graph_name="graph",
    mtk=False,
    outspec=None,
    remove_relu=False,
    prefer_same_pad_mode=False,
    disable_nhwc=False,
):
    """
	Convert traced model to TFLite,
	and save the TFLite model to file `output`.

	:param traced_module: a traced module or the file path of a traced module.
	:param output: the filename used for the saved model.
	:param data_type: data type of input
	:param graph_name: the name of the TFLite graph.
	:param mtk: if this TFLite will be run on mtk.
	:type mtk: bool
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

    set_conv_pad_perference(prefer_same_pad_mode)
    transformer_options = [
        TransformerRule.REDUCE_AXIS_AS_INPUT,
        TransformerRule.PADDING_FOR_CONV_AND_POOLING,
        TransformerRule.EXPAND_CONVRELU,
        TransformerRule.CONV_ADD_ZERO_BIAS,
        TransformerRule.DECONV_SHAPE_AS_INPUT,
        TransformerRule.DEPTHWISE_CONV_RESHAPE_WEIGHT,
        TransformerRule.RESHAPE_BIAS_TO_1DIM,
        TransformerRule.FUSE_ACTIVATION,
        TransformerRule.SLICE_PARAMS_AS_INPUTS_AND_MAKE_SQUEEZE,
        TransformerRule.RESIZE_PARAMS_AS_INPUT,
        TransformerRule.TRANSPOSE_PATTERN_AS_INPUT,
        TransformerRule.FUSE_CONV_BN,
        TransformerRule.REMOVE_IDENTITY,
        TransformerRule.REPLACE_FLATTEN_TO_RESHAPE,
        TransformerRule.PAD_WIDTH_AS_INPUT,
        TransformerRule.EXPAND_ADD_RELU,
    ]
    if mtk:
        # MTK devices only support batch_size 1
        set_platform("mtk")
        transformer_options.append(TransformerRule.DECONV_ADD_ZERO_BIAS,)
    if remove_relu:
        transformer_options.append(TransformerRule.REMOVE_TFLITE_RELU,)

    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)
    quantizer = IRQuantizer(
        require_quantize=require_quantize, param_fake_quant=param_fake_quant
    )

    if not require_quantize and tm_resolver.has_qat:
        quantizer.save_quantize_params(transformed_irgraph)
        quantizer.dump_quant_param(path=quantize_file_path)

    converter = TFLiteConverter(transformed_irgraph, graph_name, quantizer=quantizer)
    model = converter.convert(disable_nhwc)

    assert isinstance(output, str), "tflite_fpath must be string"
    with open(output, "wb") as fout:
        fout.write(model)
