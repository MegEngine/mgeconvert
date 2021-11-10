# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module,no-member

import megengine as mge
from megengine.traced_module import TracedModule

from ..backend.ir_to_caffe.caffe_converter import CaffeConverter
from ..converter_ir.ir_transform import IRTransform, TransformerRule
from ..frontend.tm_to_ir import TM_FrontEnd


def tracedmodule_to_caffe(
    traced_module,
    prototxt="out.prototxt",
    caffemodel="out.caffemodel",
    outspec=None,
    use_empty_blobs=False,
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

    irgraph = TM_FrontEnd(traced_module, outspec=outspec).resolve()
    transformer_options = [
        TransformerRule.REMOVE_DROPOUT,
        TransformerRule.REMOVE_RESHAPE_REALTED_OP,
        TransformerRule.REMOVE_UNRELATED_IROP,
        TransformerRule.ADD_FAKE_HSIGMOID_OUT,
        TransformerRule.EXPAND_CONVRELU,
    ]
    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)
    converter = CaffeConverter(transformed_irgraph, use_empty_blobs)
    converter.convert()

    assert isinstance(prototxt, str) and isinstance(
        caffemodel, str
    ), "'prototxt' and 'caffemodel' must be string"
    converter.dump(prototxt, caffemodel)
