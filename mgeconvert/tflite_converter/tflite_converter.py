# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import flatbuffers
from tqdm import tqdm

from ..mge_context import (
    TopologyNetwork,
    TransformerRule,
    optimize_for_conversion,
    set_platform,
)
from ..mge_context.mge_op import (
    Host2DeviceCopyOpr,
    MultipleDeviceTensorHolderOpr,
    SharedDeviceTensorOpr,
)
from .tflite import (  # pylint: disable=import-error
    Buffer,
    Model,
    Operator,
    OperatorCode,
    QuantizationParameters,
    SubGraph,
    Tensor,
)
from .tflite.CustomOptionsFormat import (  # pylint: disable=import-error
    CustomOptionsFormat,
)
from .tflite_op import MGE2TFLITE, get_shape_param, mge2tflite_dtype_mapping


class TFLiteConverter:
    def __init__(self, toponet, graph_name="graph"):
        assert isinstance(
            toponet, TopologyNetwork
        ), "net must be instance of TopologyNetwork"
        self.net = toponet
        self.graph_name = graph_name
        self._var2tensor = dict()  # varnode to tensor index
        self._opr_type_list = []
        self._buffer_list = []
        self._tensor_list = []
        self._operator_list = []
        self._params = {}

        # buffer size will automatically increase if needed
        self._builder = flatbuffers.Builder(1024)

        self._transformer_options = [
            TransformerRule.REDUCE_AXIS_AS_INPUT,
            TransformerRule.REMOVE_RESHAPE_INPUT,
            TransformerRule.FUSE_FOR_RELU6,
            TransformerRule.FUSE_ACTIVATION,
            TransformerRule.CONV_ADD_ZERO_BIAS,
            TransformerRule.DEPTHWISE_CONV_RESHAPE_WEIGHT,
            TransformerRule.FUSE_SOFTMAX,
            TransformerRule.DECONV_SHAPE_AS_INPUT,
            TransformerRule.MAKE_PADDING,
            TransformerRule.FUSE_ASTYPE,
            TransformerRule.TRANSPOSE_PATTERN_AS_INPUT,
            TransformerRule.FUSE_FOR_LEAKY_RELU,
            TransformerRule.EXPAND_MUL_ADD3,
            TransformerRule.EXPAND_ADD_SIGMOID,
        ]
        optimize_for_conversion(self.net, self._transformer_options)

    def convert(self, disable_nhwc=False):
        # Note the 0th entry of this array must be an empty buffer (sentinel)
        Buffer.BufferStart(self._builder)
        buffer = Buffer.BufferEnd(self._builder)
        self._buffer_list.append(buffer)

        def need_convert(mge_opr):
            is_const = [data.np_data is not None for data in mge_opr.inp_vars]
            if isinstance(
                mge_opr,
                (
                    Host2DeviceCopyOpr,
                    MultipleDeviceTensorHolderOpr,
                    SharedDeviceTensorOpr,
                ),
            ):
                return False
            return not all(is_const) or len(mge_opr.inp_vars) == 0

        for mge_opr in tqdm(self.net.all_oprs):
            last_opr = mge_opr
            if not need_convert(mge_opr):
                continue

            tfl_opr_type, tfl_options_type, tfl_options = MGE2TFLITE[type(mge_opr)](
                mge_opr, self._builder
            )
            if tfl_opr_type not in self._opr_type_list:
                self._opr_type_list.append(tfl_opr_type)

            if hasattr(mge_opr, "type") and mge_opr.type == "ConvolutionBackwardData":
                mge_opr.inp_vars = [mge_opr.inp_vars[0]] + list(
                    reversed(mge_opr.inp_vars[-2:])
                )  # shape, weight, input

            # buffer and tensor
            for var in mge_opr.inp_vars + mge_opr.out_vars:
                if var in self._var2tensor:
                    continue

                result_shape, byte_list = get_shape_param(var, mge_opr, disable_nhwc)
                var.shape = result_shape

                scale = None
                zero_point = 0
                if hasattr(var.dtype, "metadata"):
                    scale = var.dtype.metadata["mgb_dtype"]["scale"]
                    zero_point = var.dtype.metadata["mgb_dtype"].get("zero_point") or 0

                dtype = var.dtype
                if var.name in self._params.keys():
                    dtype = self._params[var.name]["dtype"]
                    scale = self._params[var.name]["scale"]
                    zero_point = self._params[var.name]["zero"]

                buffer = self.gen_buffer(byte_list)
                self._buffer_list.append(buffer)

                tfl_tensor = self.gen_tensor(
                    var.name,
                    var.shape,
                    mge2tflite_dtype_mapping[dtype],
                    len(self._buffer_list) - 1,
                    scale=scale,
                    zero_point=zero_point,
                )
                self._tensor_list.append(tfl_tensor)
                self._var2tensor[var] = len(self._tensor_list) - 1

            tfl_opr = self.gen_operator(
                mge_opr, tfl_opr_type, tfl_options_type, tfl_options
            )
            self._operator_list.append(tfl_opr)

        print("last op: {}".format(last_opr))
        out_var = last_opr.out_vars[0]
        print("dtype: {}".format(out_var.dtype))
        if hasattr(out_var.dtype, "metadata"):
            scale = out_var.dtype.metadata["mgb_dtype"]["scale"]
            zero_point = out_var.dtype.metadata["mgb_dtype"].get("zero_point") or 0
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
        Operator.OperatorStartInputsVector(self._builder, len(opr.inp_vars))
        for var in reversed(opr.inp_vars):
            self._builder.PrependInt32(self._var2tensor[var])
        inputs = self._builder.EndVector(len(opr.inp_vars))
        # outputs
        Operator.OperatorStartOutputsVector(self._builder, len(opr.out_vars))
        for var in reversed(opr.out_vars):
            self._builder.PrependInt32(self._var2tensor[var])
        outputs = self._builder.EndVector(len(opr.out_vars))

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
            OperatorCode.OperatorCodeStart(self._builder)
            OperatorCode.OperatorCodeAddBuiltinCode(self._builder, opr_type)
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
        SubGraph.SubGraphStartInputsVector(self._builder, len(self.net.input_vars))
        for var in reversed(self.net.input_vars):
            self._builder.PrependInt32(self._var2tensor[var])
        graph_inputs = self._builder.EndVector(len(self.net.input_vars))

        # outputs
        SubGraph.SubGraphStartOutputsVector(self._builder, len(self.net.output_vars))
        for var in reversed(self.net.output_vars):
            self._builder.PrependInt32(self._var2tensor[var])
        graph_outputs = self._builder.EndVector(len(self.net.output_vars))

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


def convert_to_tflite(
    mge_fpath, output="out.tflite", *, graph_name="graph", batch_size=1, mtk=False
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
    :param batch_size: batch size of TFLite model.
    :type batch_size: int
    :param mtk: if this TFLite will be run on mtk.
    :type mtk: bool
    """
    assert isinstance(mge_fpath, str), "mge_fpath must be string"
    net = TopologyNetwork(mge_fpath, prune_reshape=True)
    net.batch_size = batch_size
    if mtk:
        # MTK devices only support batch_size 1
        net.batch_size = 1
        set_platform("mtk")
    converter = TFLiteConverter(net, graph_name)
    model = converter.convert()

    assert isinstance(output, str), "tflite_fpath must be string"
    with open(output, "wb") as fout:
        fout.write(model)
