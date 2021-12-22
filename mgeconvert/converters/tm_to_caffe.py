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

from ..backend.ir_to_caffe.caffe_converter import BackEnd, CaffeConverter
from ..converter_ir.ir_quantizer import IRQuantizer
from ..converter_ir.ir_transform import IRTransform, TransformerRule
from ..frontend.tm_to_ir import TM_FrontEnd
from ..frontend.tm_to_ir.tm_utils import _update_inputs_qparams


def tracedmodule_to_caffe(
    traced_module,
    prototxt="out.prototxt",
    caffemodel="out.caffemodel",
    outspec=None,
    use_empty_blobs=False,
    input_data_type: str = None,
    input_scales: Union[float, List[float]] = None,
    input_zero_points: Union[int, List[int]] = None,
    require_quantize=False,
    param_fake_quant=False,
    split_conv_relu=False,
    fuse_bn=False,
    quantize_file_path="quant_params.json",
    convert_backend: BackEnd = BackEnd.CAFFE,
):
    """
    Convert TracedModule model to Caffe,
    and save caffe model to `prototxt` and `caffemodel`.

    :param traced_module: the file path of TracedModule model.
    :type traced_module: str
    :param prototxt: the filename used for saved model definition.
    :type prototxt: str
    :param caffemodel: the filename used for saved model weights.
    :type caffemodel: str
    :param outspec: specify the end points of the model, expect the full names of nodes.
    :type outspec: list
    """
    if isinstance(traced_module, str):
        traced_module = mge.load(traced_module)
    assert isinstance(
        traced_module, TracedModule
    ), "Input should be a traced module or a path of traced module."
    assert not require_quantize, "Caffe do not support quantize model."

    _update_inputs_qparams(
        traced_module, input_data_type, input_scales, input_zero_points
    )
    tm_resolver = TM_FrontEnd(traced_module, outspec=outspec)
    irgraph = tm_resolver.resolve()

    transformer_options = [
        TransformerRule.REMOVE_DROPOUT,
        TransformerRule.REMOVE_RESHAPE_REALTED_OP,
        TransformerRule.REMOVE_UNRELATED_IROP,
        TransformerRule.ADD_FAKE_HSIGMOID_OUT,
        TransformerRule.EXPAND_CONVRELU,
        TransformerRule.EXPAND_ADD_RELU,
    ]
    if fuse_bn:
        transformer_options += [
            TransformerRule.FUSE_LINEAR_BN,
            TransformerRule.FUSE_CONV_BN,
        ]

    if convert_backend == BackEnd.NNIE:
        transformer_options.extend(
            [TransformerRule.REMOVE_FLATTEN_BEFORE_LINEAR,]
        )

    if split_conv_relu:
        transformer_options += [TransformerRule.REMOVE_RELU]
    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)
    quantizer = IRQuantizer(
        require_quantize=require_quantize, param_fake_quant=param_fake_quant
    )

    if tm_resolver.has_qat:
        quantizer.save_quantize_params(transformed_irgraph)

    converter = CaffeConverter(
        transformed_irgraph, quantizer, use_empty_blobs, convert_backend
    )
    converter.convert()

    if tm_resolver.has_qat:
        quantizer.dump_quant_param(path=quantize_file_path)

    assert isinstance(prototxt, str) and isinstance(
        caffemodel, str
    ), "'prototxt' and 'caffemodel' must be string"
    converter.dump(prototxt, caffemodel)
