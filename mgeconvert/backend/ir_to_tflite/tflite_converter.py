# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error
import flatbuffers
from tqdm import tqdm

from ...converter_ir.ir_graph import IRGraph
from ...converter_ir.ir_op import (
    ConstantOpr,
    LinspaceOpr,
    MultipleDeviceTensorHolderOpr,
    SharedDeviceTensorOpr,
)
from .tflite import (
    Buffer,
    Model,
    Operator,
    OperatorCode,
    QuantizationParameters,
    SubGraph,
    Tensor,
)
from .tflite.CustomOptionsFormat import CustomOptionsFormat
from .tflite_op import (
    MGE2TFLITE,
    get_shape_param,
    mge2tflite_dtype_mapping,
    set_quantization,
    set_tensor_format,
)


class TFLiteConverter:
    def __init__(self, net, graph_name="graph", quantizer=None):
        assert isinstance(net, IRGraph), "net must be instance of IRGraph"
        self.net = net
        self.graph_name = graph_name
        self._var2tensor = dict()  # varnode to tensor index
        self._opr_type_list = []
        self._buffer_list = []
        self._tensor_list = []
        self._operator_list = []
        self.quantizer = quantizer

        # buffer size will automatically increase if needed
        self._builder = flatbuffers.Builder(1024)
        set_quantization(require_quantize=quantizer.require_quantize)

    def convert(self, disable_nhwc=False):
        # Note the 0th entry of this array must be an empty buffer (sentinel)
        if disable_nhwc:
            set_tensor_format("nchw")
        else:
            set_tensor_format("nhwc")
        Buffer.BufferStart(self._builder)
        buffer = Buffer.BufferEnd(self._builder)
        self._buffer_list.append(buffer)

        def need_convert(mge_opr):
            if isinstance(
                mge_opr,
                (
                    ConstantOpr,
                    LinspaceOpr,
                    MultipleDeviceTensorHolderOpr,
                    SharedDeviceTensorOpr,
                ),
            ):
                return False
            is_const = [data.np_data is not None for data in mge_opr.inp_tensors]
            return not all(is_const) and len(mge_opr.inp_tensors) > 0

        for mge_opr in tqdm(self.net.all_oprs):
            last_opr = mge_opr
            if not need_convert(mge_opr):
                continue

            tfl_opr_type, tfl_options_type, tfl_options = MGE2TFLITE[type(mge_opr)](
                mge_opr, self._builder
            )
            if tfl_opr_type not in self._opr_type_list:
                self._opr_type_list.append(tfl_opr_type)

            # buffer and tensor
            for tensor in mge_opr.inp_tensors + mge_opr.out_tensors:
                if tensor in self._var2tensor:
                    continue
                result_shape, byte_list = get_shape_param(
                    tensor, mge_opr, self.quantizer, disable_nhwc=disable_nhwc
                )

                scale = None
                zero_point = 0

                if self.quantizer.require_quantize:
                    has_qparams = False
                    if hasattr(tensor, "scale") and tensor.scale is not None:
                        scale = tensor.scale
                        has_qparams = True
                    if hasattr(tensor, "zero_point") and tensor.zero_point is not None:
                        zero_point = int(tensor.zero_point)
                    dtype = tensor.q_dtype if has_qparams else tensor.dtype
                    from megengine.core.tensor.dtype import (  # pylint: disable=import-outside-toplevel,no-name-in-module
                        QuantDtypeMeta,
                    )

                    if isinstance(dtype, QuantDtypeMeta):
                        dtype = dtype.name
                else:
                    dtype = tensor.dtype

                buffer = self.gen_buffer(byte_list)
                self._buffer_list.append(buffer)
                tfl_tensor = self.gen_tensor(
                    tensor.name,
                    result_shape,
                    mge2tflite_dtype_mapping[dtype],
                    len(self._buffer_list) - 1,
                    scale=scale,
                    zero_point=int(zero_point),
                )
                self._tensor_list.append(tfl_tensor)
                self._var2tensor[tensor] = len(self._tensor_list) - 1

            tfl_opr = self.gen_operator(
                mge_opr, tfl_opr_type, tfl_options_type, tfl_options
            )
            self._operator_list.append(tfl_opr)

        print("last op: {}".format(last_opr))
        out_tensor = last_opr.out_tensors[0]
        print("dtype: {}".format(out_tensor.dtype))
        if hasattr(out_tensor.dtype, "metadata"):
            scale = out_tensor.dtype.metadata["mgb_dtype"]["scale"]
            zero_point = out_tensor.dtype.metadata["mgb_dtype"].get("zero_point") or 0
            print("scale: {}, zero point: {}".format(scale, zero_point))

        return self.get_model()

    def gen_buffer(self, byte_list):
        if not byte_list:
            Buffer.BufferStart(self._builder)
            buffer = Buffer.BufferEnd(self._builder)
        else:
            Buffer.BufferStartDataVector(self._builder, len(byte_list))
            for i in reversed(byte_list):
                self._builder.PrependByte(i)
            datas = self._builder.EndVector(len(byte_list))
            Buffer.BufferStart(self._builder)
            Buffer.BufferAddData(self._builder, datas)
            buffer = Buffer.BufferEnd(self._builder)
        return buffer

    def gen_tensor(
        self, tensor_name, result_shape, dtype, buffer_idx, scale=None, zero_point=0
    ):
        name = self._builder.CreateString(tensor_name)
        Tensor.TensorStartShapeVector(self._builder, len(result_shape))
        for i in reversed(result_shape):
            self._builder.PrependInt32(i)
        shape = self._builder.EndVector(len(result_shape))
        if scale:
            QuantizationParameters.QuantizationParametersStartScaleVector(
                self._builder, 1
            )
            self._builder.PrependFloat32(scale)
            scales = self._builder.EndVector(1)
            QuantizationParameters.QuantizationParametersStartZeroPointVector(
                self._builder, 1
            )
            self._builder.PrependInt64(zero_point)
            zero_points = self._builder.EndVector(1)
            QuantizationParameters.QuantizationParametersStart(self._builder)
            QuantizationParameters.QuantizationParametersAddScale(self._builder, scales)
            QuantizationParameters.QuantizationParametersAddZeroPoint(
                self._builder, zero_points
            )
            qp = QuantizationParameters.QuantizationParametersEnd(self._builder)
        Tensor.TensorStart(self._builder)
        Tensor.TensorAddName(self._builder, name)
        Tensor.TensorAddShape(self._builder, shape)
        Tensor.TensorAddType(self._builder, dtype)
        Tensor.TensorAddBuffer(self._builder, buffer_idx)
        if scale:
            Tensor.TensorAddQuantization(self._builder, qp)
        tensor = Tensor.TensorEnd(self._builder)
        return tensor

    def gen_operator(self, opr, opr_type, options_type, options):
        # opcode_index
        opcode_index = self._opr_type_list.index(opr_type)
        # inputs
        Operator.OperatorStartInputsVector(self._builder, len(opr.inp_tensors))
        for var in reversed(opr.inp_tensors):
            self._builder.PrependInt32(self._var2tensor[var])
        inputs = self._builder.EndVector(len(opr.inp_tensors))
        # outputs
        Operator.OperatorStartOutputsVector(self._builder, len(opr.out_tensors))
        for var in reversed(opr.out_tensors):
            self._builder.PrependInt32(self._var2tensor[var])
        outputs = self._builder.EndVector(len(opr.out_tensors))

        custom_options = None
        builtin_options = None
        if options:
            if isinstance(options, bytes):  # custom_options
                Operator.OperatorStartCustomOptionsVector(self._builder, len(options))
                for i in reversed(options):
                    self._builder.PrependByte(i)
                custom_options = self._builder.EndVector(len(options))
            else:  # builtin_options
                builtin_options = options

        Operator.OperatorStart(self._builder)
        Operator.OperatorAddOpcodeIndex(self._builder, opcode_index)
        Operator.OperatorAddInputs(self._builder, inputs)
        Operator.OperatorAddOutputs(self._builder, outputs)
        if custom_options:
            Operator.OperatorAddCustomOptions(self._builder, custom_options)
            Operator.OperatorAddCustomOptionsFormat(
                self._builder, CustomOptionsFormat.FLEXBUFFERS
            )
        elif builtin_options:
            Operator.OperatorAddBuiltinOptionsType(self._builder, options_type)
            Operator.OperatorAddBuiltinOptions(self._builder, builtin_options)
        operator = Operator.OperatorEnd(self._builder)
        return operator

    def get_version(self):
        return 3

    def get_description(self):
        description = self._builder.CreateString("Converted by MegEngine")
        return description

    def get_operator_codes(self):
        operator_codes_list = []
        for opr_type in self._opr_type_list:
            is_custom = not isinstance(opr_type, int)
            if is_custom:
                custom_code = self._builder.CreateString(opr_type.code)
                opr_type = opr_type.type
            OperatorCode.OperatorCodeStart(self._builder)
            OperatorCode.OperatorCodeAddBuiltinCode(self._builder, opr_type)
            if is_custom:
                OperatorCode.OperatorCodeAddCustomCode(self._builder, custom_code)
            operator_code = OperatorCode.OperatorCodeEnd(self._builder)
            operator_codes_list.append(operator_code)

        Model.ModelStartOperatorCodesVector(self._builder, len(operator_codes_list))
        for i in reversed(operator_codes_list):
            self._builder.PrependUOffsetTRelative(i)
        operator_codes = self._builder.EndVector(len(operator_codes_list))
        return operator_codes

    def get_subgraphs(self):
        # only support one subgraph now
        subgraphs_list = []

        # tensors
        SubGraph.SubGraphStartTensorsVector(self._builder, len(self._tensor_list))
        for tensor in reversed(self._tensor_list):
            self._builder.PrependUOffsetTRelative(tensor)
        tensors = self._builder.EndVector(len(self._tensor_list))

        # inputs
        SubGraph.SubGraphStartInputsVector(self._builder, len(self.net.graph_inputs))
        for var in reversed(self.net.graph_inputs):
            self._builder.PrependInt32(self._var2tensor[var])
        graph_inputs = self._builder.EndVector(len(self.net.graph_inputs))

        # outputs
        SubGraph.SubGraphStartOutputsVector(self._builder, len(self.net.graph_outputs))
        for var in reversed(self.net.graph_outputs):
            self._builder.PrependInt32(self._var2tensor[var])
        graph_outputs = self._builder.EndVector(len(self.net.graph_outputs))

        # operators
        SubGraph.SubGraphStartOperatorsVector(self._builder, len(self._operator_list))
        for operator in reversed(self._operator_list):
            self._builder.PrependUOffsetTRelative(operator)
        operators = self._builder.EndVector(len(self._operator_list))

        # name
        sub_graph_name = self._builder.CreateString("graph0")

        SubGraph.SubGraphStart(self._builder)
        SubGraph.SubGraphAddTensors(self._builder, tensors)
        SubGraph.SubGraphAddInputs(self._builder, graph_inputs)
        SubGraph.SubGraphAddOutputs(self._builder, graph_outputs)
        SubGraph.SubGraphAddOperators(self._builder, operators)
        SubGraph.SubGraphAddName(self._builder, sub_graph_name)
        subgraph = SubGraph.SubGraphEnd(self._builder)
        subgraphs_list.append(subgraph)

        Model.ModelStartSubgraphsVector(self._builder, len(subgraphs_list))
        for i in reversed(subgraphs_list):
            self._builder.PrependUOffsetTRelative(i)
        subgraphs = self._builder.EndVector(len(subgraphs_list))
        return subgraphs

    def get_buffers(self):
        Model.ModelStartBuffersVector(self._builder, len(self._buffer_list))
        for i in reversed(self._buffer_list):
            self._builder.PrependUOffsetTRelative(i)
        buffers = self._builder.EndVector(len(self._buffer_list))
        return buffers

    def get_model(self):
        version = self.get_version()
        operator_codes = self.get_operator_codes()
        subgraphs = self.get_subgraphs()
        description = self.get_description()
        buffers = self.get_buffers()

        Model.ModelStart(self._builder)
        Model.ModelAddVersion(self._builder, version)
        Model.ModelAddOperatorCodes(self._builder, operator_codes)
        Model.ModelAddSubgraphs(self._builder, subgraphs)
        Model.ModelAddDescription(self._builder, description)
        Model.ModelAddBuffers(self._builder, buffers)

        model = Model.ModelEnd(self._builder)
        self._builder.Finish(model, "TFL3".encode())
        return self._builder.Output()
