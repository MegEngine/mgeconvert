import flatbuffers
import numpy as np
from mgeconvert.tflite_converter import tflite
from mgeconvert.tflite_converter.tflite import (
    Buffer,
    Conv2DOptions,
    Model,
    NegOptions,
    Operator,
    OperatorCode,
    QuantizationParameters,
    SubGraph,
    Tensor,
    TensorType,
)
from mgeconvert.tflite_converter.tflite.BuiltinOperator import BuiltinOperator
from mgeconvert.tflite_converter.tflite.BuiltinOptions import BuiltinOptions
from mgeconvert.tflite_converter.tflite.CustomOptionsFormat import CustomOptionsFormat

buffer_list = []
# buffer size will automatically increase if needed
builder = flatbuffers.Builder(1024)

# Note the 0th entry of this array must be an empty buffer (sentinel)
Buffer.BufferStart(builder)
buffer = Buffer.BufferEnd(builder)
buffer_list.append(buffer)

inp_shape = (2, 2, 2, 2)

tensor_list = []
for name in ["x", "z"]:
    byte_list = None

    # tensor buffer
    Buffer.BufferStart(builder)
    buffer = Buffer.BufferEnd(builder)
    buffer_list.append(buffer)

    # tensor
    tname = builder.CreateString(name)

    Tensor.TensorStartShapeVector(builder, len(inp_shape))
    for i in reversed(inp_shape):
        builder.PrependInt32(i)
    shape = builder.EndVector(len(inp_shape))

    Tensor.TensorStart(builder)
    Tensor.TensorAddName(builder, tname)
    Tensor.TensorAddShape(builder, shape)
    Tensor.TensorAddType(builder, TensorType.TensorType.INT32)
    Tensor.TensorAddBuffer(builder, len(buffer_list) - 1)
    tensor = Tensor.TensorEnd(builder)
    tensor_list.append(tensor)

opcode_index = 0
operator_list = []
# inputs
Operator.OperatorStartInputsVector(builder, 1)
builder.PrependInt32(0)
inputs = builder.EndVector(1)
# outputs
Operator.OperatorStartOutputsVector(builder, 1)
builder.PrependInt32(1)
outputs = builder.EndVector(1)

# options
NegOptions.NegOptionsStart(builder)
builtin_options = NegOptions.NegOptionsEnd(builder)

Operator.OperatorStart(builder)
Operator.OperatorAddOpcodeIndex(builder, opcode_index)
Operator.OperatorAddInputs(builder, inputs)
Operator.OperatorAddOutputs(builder, outputs)
Operator.OperatorAddBuiltinOptionsType(builder, BuiltinOptions.NegOptions)
Operator.OperatorAddBuiltinOptions(builder, builtin_options)
operator = Operator.OperatorEnd(builder)
operator_list.append(operator)

version = 3
description = builder.CreateString("converted by MgeEngine")

operator_codes_list = []
OperatorCode.OperatorCodeStart(builder)
OperatorCode.OperatorCodeAddBuiltinCode(builder, BuiltinOperator.NEG)
operator_code = OperatorCode.OperatorCodeEnd(builder)
operator_codes_list.append(operator_code)

Model.ModelStartOperatorCodesVector(builder, len(operator_codes_list))
for i in reversed(operator_codes_list):
    builder.PrependUOffsetTRelative(i)
operator_codes = builder.EndVector(len(operator_codes_list))

subgraphs_list = []
# tensors
SubGraph.SubGraphStartTensorsVector(builder, len(tensor_list))
for tensor in reversed(tensor_list):
    builder.PrependUOffsetTRelative(tensor)
tensors = builder.EndVector(len(tensor_list))

# inputs
SubGraph.SubGraphStartInputsVector(builder, 1)
builder.PrependInt32(0)
graph_inputs = builder.EndVector(1)

# outputs
SubGraph.SubGraphStartOutputsVector(builder, 1)
builder.PrependInt32(1)
graph_outputs = builder.EndVector(1)

# operators
SubGraph.SubGraphStartOperatorsVector(builder, len(operator_list))
for operator in reversed(operator_list):
    builder.PrependUOffsetTRelative(operator)
operators = builder.EndVector(len(operator_list))

# name
sub_graph_name = builder.CreateString("graph0")

SubGraph.SubGraphStart(builder)
SubGraph.SubGraphAddTensors(builder, tensors)
SubGraph.SubGraphAddInputs(builder, graph_inputs)
SubGraph.SubGraphAddOutputs(builder, graph_outputs)
SubGraph.SubGraphAddOperators(builder, operators)
SubGraph.SubGraphAddName(builder, sub_graph_name)
subgraph = SubGraph.SubGraphEnd(builder)
subgraphs_list.append(subgraph)

Model.ModelStartSubgraphsVector(builder, len(subgraphs_list))
for i in reversed(subgraphs_list):
    builder.PrependUOffsetTRelative(i)
subgraphs = builder.EndVector(len(subgraphs_list))

Model.ModelStartBuffersVector(builder, len(buffer_list))
for i in reversed(buffer_list):
    builder.PrependUOffsetTRelative(i)
buffers = builder.EndVector(len(buffer_list))

Model.ModelStart(builder)
Model.ModelAddVersion(builder, version)
Model.ModelAddOperatorCodes(builder, operator_codes)
Model.ModelAddSubgraphs(builder, subgraphs)
Model.ModelAddDescription(builder, description)
Model.ModelAddBuffers(builder, buffers)
model = Model.ModelEnd(builder)
builder.Finish(model, tflite.GetFileIdentifier())

with open("out.tflite", "wb") as f:
    f.write(builder.Output())

# # validate model file
# from tensorflow.lite.python import interpreter
# model = interpreter.Interpreter(model_path="./out.tflite")
# model.allocate_tensors()
# input = model.tensor(model.get_input_details()[0]["index"])
# output = model.tensor(model.get_output_details()[0]["index"])
# input().fill(3.)
# model.invoke()
# print(output())
