# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..backend.ir_to_tflite import TFLiteConverter, set_platform
from ..converter_ir.ir_quantizer import IRQuantizer
from ..converter_ir.ir_transform import (
    IRTransform,
    TransformerRule,
    set_conv_pad_perference,
)
from ..frontend.mge_to_ir import MGE_FrontEnd


def mge_to_tflite(
    mge_fpath,
    output="out.tflite",
    *,
    graph_name="graph",
    mtk=False,
    disable_nhwc=False,
    outspec=None,
    prefer_same_pad_mode=False,
):
    """
    Convert megengine model to TFLite,
    and save the TFLite model to file `output`.

    :param mge_fpath: the file path of megengine model.
    :type fpath: str
    :param output: the filename used for the saved model.
    :type output: str
    :param graph_name: the name of the TFLite graph.
    :type graph_name: str
    :param mtk: if this TFLite will be run on mtk.
    :type mtk: bool
    :param outspec: the names of end points of the model.
    :type outspec: list
    """
    assert isinstance(mge_fpath, str), "mge_fpath must be string"
    irgraph = MGE_FrontEnd(mge_fpath, outspec=outspec).resolve()
    set_conv_pad_perference(prefer_same_pad_mode)
    transformer_options = [
        TransformerRule.REDUCE_AXIS_AS_INPUT,
        TransformerRule.PADDING_FOR_CONV_AND_POOLING,
        TransformerRule.CONV_ADD_ZERO_BIAS,
        TransformerRule.DECONV_SHAPE_AS_INPUT,
        TransformerRule.DEPTHWISE_CONV_RESHAPE_WEIGHT,
        TransformerRule.RESHAPE_BIAS_TO_1DIM,
        TransformerRule.FUSE_ACTIVATION,
        TransformerRule.SLICE_PARAMS_AS_INPUTS_AND_MAKE_SQUEEZE,
        TransformerRule.RESIZE_PARAMS_AS_INPUT,
        TransformerRule.TRANSPOSE_PATTERN_AS_INPUT,
        TransformerRule.REMOVE_RESHAPE_INPUT,
        TransformerRule.FUSE_SOFTMAX,
        TransformerRule.EXPAND_MUL_ADD3,
        TransformerRule.FUSE_FOR_CONV_BIAS,
        TransformerRule.FUSE_FOR_LEAKY_RELU,
        TransformerRule.REMOVE_RESHAPE_INPUT,
        TransformerRule.FUSE_CONV_BN,
        TransformerRule.REMOVE_IDENTITY,
        TransformerRule.EXPAND_ADD_SIGMOID,
    ]
    if mtk:
        # MTK devices only support batch_size 1
        set_platform("mtk")
        transformer_options.append(TransformerRule.DECONV_ADD_ZERO_BIAS,)
        transformer_options.append(TransformerRule.FUSE_FOR_DECONV_BIAS,)

    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)

    quantizer = IRQuantizer(require_quantize=False)

    converter = TFLiteConverter(transformed_irgraph, graph_name, quantizer=quantizer)
    model = converter.convert(disable_nhwc=disable_nhwc)

    assert isinstance(output, str), "tflite_fpath must be string"
    with open(output, "wb") as fout:
        fout.write(model)
