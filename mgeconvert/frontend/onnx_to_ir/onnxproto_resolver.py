# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import numpy as np
import onnx

from ...converter_ir.ir_tensor import IRTensor

onnx2np_dtype_mapping = {
    # pylint: disable=no-member
    None: None,
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.INT16: np.int16,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.UINT16: np.uint16,
}


class ONNXProtoResolver:
    def __init__(self, onnx_model):
        self.onnx_model = onnx_model

    def get_ir_tensor_from_valueinfo(self, value_info, owner_opr=None):
        type_proto = value_info.type
        dtype = onnx2np_dtype_mapping[type_proto.tensor_type.elem_type]
        shape = []
        for n in type_proto.tensor_type.shape.dim:
            shape.append(n.dim_value)
        return IRTensor(value_info.name, shape, dtype, owner_opr=owner_opr)

    def get_ir_tensor_from_initializer(self, initializer, owner_opr=None):
        dtype = onnx2np_dtype_mapping[initializer.data_type]
        shape = []
        if initializer.dims == [0]:
            return IRTensor(
                initializer.name, shape, dtype, np_data=None, owner_opr=owner_opr
            )
        for n in initializer.dims:
            shape.append(n)
        if len(initializer.raw_data) != 0:
            np_data = initializer.raw_data
        elif len(initializer.float_data) != 0:
            np_data = np.array(initializer.float_data, dtype=np.float32).tobytes()
        elif len(initializer.int32_data) != 0:
            np_data = np.array(initializer.int32_data, dtype=np.int32).tobytes()
        elif len(initializer.int64_data) != 0:
            np_data = np.array(initializer.int64_data, dtype=np.int64).tobytes()
        else:
            raise AttributeError(f"Invalid Data Type : {initializer}")
        return IRTensor(
            initializer.name, shape, dtype, np_data=np_data, owner_opr=owner_opr
        )

    def get_ir_tensor_from_name(self, name, owner_opr=None):
        for tensor in self.onnx_model.graph.initializer:
            if tensor.name == name:
                return self.get_ir_tensor_from_initializer(tensor, owner_opr)
        for input in self.onnx_model.graph.input:
            if input.name == name:
                return self.get_ir_tensor_from_valueinfo(input)
        for output in self.onnx_model.graph.output:
            if output.name == name:
                return self.get_ir_tensor_from_valueinfo(output)
        for v in self.onnx_model.graph.value_info:
            if v.name == name:
                return self.get_ir_tensor_from_valueinfo(v)
        return IRTensor(name, shape=[], dtype=None, owner_opr=owner_opr)
