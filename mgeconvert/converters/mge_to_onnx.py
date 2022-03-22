# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..backend.ir_to_onnx import OnnxConverter
from ..converter_ir.ir_transform import IRTransform, TransformerRule
from ..frontend.mge_to_ir import MGE_FrontEnd


def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            "Model with ir_version below 4 requires to include initilizer in graph input"
        )
        return model

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model


def mge_to_onnx(
    mge_fpath, output="out.onnx", *, graph_name="graph", opset=8, outspec=None
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
    :param outspec: the names of end points of the model.
    :type outspec: list
    """
    assert isinstance(mge_fpath, str), "mge_fpath must be string"
    irgraph = MGE_FrontEnd(mge_fpath, outspec=outspec).resolve()
    transformer_options = [
        TransformerRule.FUSE_SOFTMAX,
        TransformerRule.EXPAND_MUL_ADD3,
        TransformerRule.EXPAND_ADD_SIGMOID,
    ]
    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)
    converter = OnnxConverter(transformed_irgraph, opset, graph_name)
    model = converter.convert()
    model = remove_initializer_from_input(model)

    assert isinstance(output, str), "onnx_fpath must be string"
    with open(output, "wb") as fout:
        fout.write(model.SerializeToString())
