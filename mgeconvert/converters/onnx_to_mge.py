# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ..backend.ir_to_mge import MGEConverter
from ..frontend.onnx_to_ir import ONNX_FrontEnd


def onnx_to_mge(onnx_fpath, output="out.mge"):
    """
    Convert ONNX model to MGE model and save the MGE model to file `output`.
    :param onnx_fpath: the file path of onnx model.
    :type fpath: str
    :param output: the filename used for the saved MGE model.
    :type output: str
    """
    assert isinstance(onnx_fpath, str), "onnx_fpath must be string"
    ir_graph = ONNX_FrontEnd(onnx_fpath).resolve()
    assert isinstance(output, str), "mge_fpath must be string"
    converter = MGEConverter(ir_graph)
    converter.dump_mge_model(output)
