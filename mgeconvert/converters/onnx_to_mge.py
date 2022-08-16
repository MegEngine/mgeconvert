# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import time

from ..backend.ir_to_mge import MGEConverter
from ..converter_ir.ir_transform import IRTransform, TransformerRule
from ..frontend.onnx_to_ir import ONNX_FrontEnd


def onnx_to_mge(
    onnx_fpath,
    output="out.mge",
    *,
    optimize_for_inference=False,
    frozen_input_shape=False,
):
    """
    Convert ONNX model to MGE model and save the MGE model to file `output`.
    :param onnx_fpath: the file path of onnx model.
    :type fpath: str
    :param output: the filename used for the saved MGE model.
    :type output: str
    """
    assert isinstance(onnx_fpath, str), "onnx_fpath must be string"
    assert isinstance(output, str), "mge_fpath must be string"

    seed = int(time.time())
    front = ONNX_FrontEnd(onnx_fpath, dynamic_input_shape=not frozen_input_shape)
    ir_graph = front.resolve()
    onnx_res = front.eval(seed)
    transformer_options = [
        TransformerRule.FC_NO_TRANS,
    ]
    transformer = IRTransform(transformer_options)
    ir_graph = transformer.transform(ir_graph)
    converter = MGEConverter(ir_graph)
    mge_res = converter.eval(seed)

    # check forwarding result
    for onnx_data, mge_data in zip(onnx_res, mge_res):
        if onnx_data.dtype in ["uint8", "int8"]:
            eps = 1
        else:
            eps = 1e-3
        onnx_data_f = onnx_data.flatten()
        mge_data_f = mge_data.numpy().flatten()
        for i, (i1, i2) in enumerate(zip(onnx_data_f, mge_data_f)):
            assert (
                abs(float(i1) - float(i2)) / (abs(float(i1)) + abs(float(i2) + 1e-9))
                <= eps
                or abs(float(i1) - float(i2)) <= eps
            ), f"Forward Result of ONNX and Mge Mismatch {i1}(ONNX) vs {i2}(Mge) with model at index {i}"

    converter.dump_mge_model(output, optimize_for_inference)
