# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import time

from ..backend.ir_to_mge import MGEConverter
from ..frontend.onnx_to_ir import ONNX_FrontEnd


def onnx_to_tracedmodule(onnx_fpath, output="out.mge"):
    """
    Convert ONNX model to MGE model and save the MGE model to file `output`.
    :param onnx_fpath: the file path of onnx model.
    :type fpath: str
    :param output: the filename used for the saved MGE model.
    :type output: str
    """
    seed = int(time.time())
    assert isinstance(onnx_fpath, str), "onnx_fpath must be string"
    assert isinstance(output, str), "mge_fpath must be string"

    front = ONNX_FrontEnd(onnx_fpath)
    ir_graph = front.resolve()
    onnx_res = front.eval(seed)

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
                abs(float(i1) - float(i2)) <= eps
            ), f"Forward Result of ONNX and Mge Mismatch {i1}(ONNX) vs {i2}(Mge) with model at index {i}"

    # finally dump traced module
    converter.dump_tm_model(output)
