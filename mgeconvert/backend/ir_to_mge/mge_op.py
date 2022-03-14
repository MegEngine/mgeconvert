# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np

from ...converter_ir.ir_op import (
    AddOpr,
    AvgPool2dOpr,
    ClipOpr,
    ConcatOpr,
    Conv2dOpr,
    DropoutOpr,
    FlattenOpr,
    GatherOpr,
    GetSubTensorOpr,
    LstmOpr,
    MatMulOpr,
    MaxPool2dOpr,
    MulOpr,
    ReluOpr,
    ReshapeOpr,
    ResizeOpr,
    SigmoidOpr,
    SoftmaxOpr,
    TransposeOpr,
    TypeCvtOpr,
)

mge_version = mge.__version__

ONNX2MGE = {}


def _register_op(*oprs):
    def callback(impl):
        for opr in oprs:
            ONNX2MGE[opr] = impl
        return impl

    return callback


PARAMEXTRACT = {}


def _register_param_extract(*oprs):
    def callback(impl):
        for opr in oprs:
            PARAMEXTRACT[opr] = impl
        return impl

    return callback


MGE_MODULE_SUPPORT = {}


def _register_mge_module(*oprs):
    def callback(impl):
        for opr in oprs:
            MGE_MODULE_SUPPORT[opr] = impl
        return impl

    return callback


def convert_onnx_padding_to_mge_padding(onnx_padding):
    l = len(onnx_padding)
    assert l % 2 == 0, "padding should be odd number"
    if l == 2:
        return onnx_padding
    else:
        pad_ndim = int(l / 2)
        padding = [0] * pad_ndim
        for i in range(pad_ndim):
            assert (
                onnx_padding[i] == onnx_padding[i + pad_ndim]
            ), f"MegEngine only Supports padding same size on both sides : {onnx_padding}"

            padding[i] = onnx_padding[i]
        return tuple(padding)


class OperatorBaseConverter:

    __opr_type__ = "OperatorBaseConverter"

    def __init__(self, opr, param=None, quantizer=None):
        """
        :param opr: the operator that converter converts.
        :type opr: subclass of :class:`.MgeOpr`
        """
        self._opr = opr
        self.param = param
        self.quantizer = quantizer

    def get_inputs(
        self,
        map_ir_tensor_2_mge_tensor,
        input_names,
        input_dtypes,
        input_shapes,
        input_datas,
    ):
        """
        Get Mge Tensors from a map of {tensor_name_in_onnx, mge_tensor} or construct a mge tensor from dtype, shape and datas
        """
        inp = []
        for name, dtype, shape, data in zip(
            input_names, input_dtypes, input_shapes, input_datas
        ):
            if name in map_ir_tensor_2_mge_tensor:
                inp.append(map_ir_tensor_2_mge_tensor[name])
                if shape is not None:
                    mge_shape = map_ir_tensor_2_mge_tensor[name].shape.numpy()
                    assert (
                        shape == mge_shape
                    ).all(), f"ONNX shape Infer mismatch with Mge : {shape}(ONNX) vs {mge_shape}Mge"
            else:
                assert (
                    data is not None
                ), "This Tensor should be parameter given by model"
                np_data = np.frombuffer(data, dtype=dtype).reshape(shape)
                x = mge.tensor(np_data)
                map_ir_tensor_2_mge_tensor[name] = x
                inp.append(x)
        return inp

    def set_outputs(self, map_ir_tensor_2_mge_tensor, out_tensors, output_names):
        """
        Set Mge Tensors to a map of {tensor_name_in_onnx, mge_tensor}
        """
        if not isinstance(out_tensors, list):
            out_tensors = [out_tensors]
        for x, name in zip(out_tensors, output_names):
            map_ir_tensor_2_mge_tensor[name] = x

    def forward(self, inps):
        """
        Forward method override by Mge Opr
        """
        return inps

    def check_valid(self, inputs, outputs):
        """
        Check whether inputs and outputs is valid
        """

    def convert(
        self,
        inputs,
        dtypes,
        shapes,
        datas,
        outputs,
        map_ir_tensor_2_mge_tensor,
        map_op_name_2_mge_module,
    ):
        # pylint: disable=W0612,W0613
        """
        Do check, find, forward and set with Mge Tensors
        """
        self.check_valid(inputs, outputs)
        inps = self.get_inputs(
            map_ir_tensor_2_mge_tensor, inputs, dtypes, shapes, datas
        )
        x = self.forward(inps)
        self.set_outputs(map_ir_tensor_2_mge_tensor, x, outputs)
        return x


class SISOConvert(OperatorBaseConverter):
    def check_valid(self, inputs, outputs):
        assert len(inputs) == 1, "Length of inputs should be 1"
        assert len(outputs) == 1, "Length of outptus should be 1"


class TISOConvert(OperatorBaseConverter):
    def check_valid(self, inputs, outputs):
        assert len(inputs) == 2, "Length of inputs should be 2"
        assert len(outputs) == 1, "Length of outptus should be 1"


@_register_op(MulOpr)
class MulConverter(TISOConvert):
    def forward(self, inps):
        return F.mul(inps[0], inps[1])


@_register_op(AddOpr)
class AddConverter(TISOConvert):
    def forward(self, inps):
        return F.add(inps[0], inps[1])


@_register_op(SigmoidOpr)
class SigmoidConverter(SISOConvert):
    def forward(self, inps):
        return F.nn.sigmoid(inps[0])


@_register_op(ReluOpr)
class ReluConverter(SISOConvert):
    def forward(self, inps):
        return F.nn.relu(inps[0])


@_register_param_extract(ReshapeOpr)
class ReshapeExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        if self._opr.allowzero:
            for dim in self._opr.out_shape:
                assert dim != 0, "MegEngine did not support empty shape when Reshape"
        param = {}
        param["out_shape"] = self._opr.out_shape
        return param


@_register_op(ReshapeOpr)
class ReshapeConverter(SISOConvert):
    def forward(self, inps):
        target_shape = self.param["out_shape"]
        if 0 not in target_shape and -1 not in target_shape:
            return F.reshape(inps[0], target_shape)
        else:
            tshape = []
            inp_shape = inps[0].shape.numpy()
            for i, k in enumerate(target_shape):
                if k == 0:
                    tshape.append(inp_shape[i])
                else:
                    tshape.append(k)
            return F.reshape(inps[0], tshape)


@_register_param_extract(GetSubTensorOpr)
class SubTensorExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        begin_param = self._opr.begin_params
        end_param = self._opr.end_params
        step_param = self._opr.step_params
        axis_param = self._opr.axis
        slices = [slice(None, None, None)] * (max(axis_param) + 1)
        for i, axis in enumerate(axis_param):
            try:
                slices[axis] = slice(begin_param[i], end_param[i], step_param[i])
            except IndexError:
                slices[axis] = slice(begin_param[i], end_param[i], None)
        return {"index": tuple(slices)}


@_register_op(GetSubTensorOpr)
class SubtensorConverter(SISOConvert):
    def forward(self, inps):
        return inps[0][self.param["index"]]


@_register_param_extract(TransposeOpr)
class DimshuffleExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        pattern = [str(i) for i in self._opr.pattern]
        return {"perm": pattern}


@_register_op(TransposeOpr)
class DimshuffleConverter(SISOConvert):
    def forward(self, inps):
        pattern = [int(i) for i in self.param["perm"]]
        return F.transpose(inps[0], pattern)


@_register_param_extract(TypeCvtOpr)
class TypeCvtExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        return {"out_dtype": self._opr.out_dtype}


@_register_op(TypeCvtOpr)
class TypeCvtOprConverter(SISOConvert):
    def forward(self, inps):
        return inps[0].astype(self.param["out_dtype"])


@_register_param_extract(Conv2dOpr)
class Conv2dExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        opr = self._opr
        param = {}
        param["stride"] = opr.stride
        param["dilation"] = opr.dilation
        param["groups"] = opr.groups
        assert (
            opr.auto_pad == "NOTSET"
        ), "ONNX To MegEngine Convert Only supports NOTSET pad mode in Conv2d"

        param["padding"] = convert_onnx_padding_to_mge_padding(opr.padding)

        return param


@_register_op(Conv2dOpr)
class Conv2dOprConverter(OperatorBaseConverter):
    def forward(self, inps):
        src = inps[0]
        weight = inps[1]
        try:
            bias = inps[2]
        except IndexError:
            bias = None

        if bias is not None:
            if bias.shape.ndim == 3:
                bias = F.expand_dims(bias, axis=0)
            elif bias.shape.ndim == 1:
                bias = F.expand_dims(bias, axis=[0, 2, 3])
            else:
                raise Exception(f"Invalid Conv2d bias's shape {bias.shape}")

        if self.param["groups"] != 1:
            groups = self.param["groups"]
            IC = src.shape.numpy()[1]
            OC = weight.shape.numpy()[0]
            FH = weight.shape.numpy()[2]
            FW = weight.shape.numpy()[3]
            target_shape = [groups, int(OC / groups), int(IC / groups), FH, FW]
            weight = F.reshape(weight, target_shape)

        return F.conv2d(
            src,
            weight,
            bias,
            stride=self.param["stride"],
            padding=self.param["padding"],
            dilation=self.param["dilation"],
            groups=self.param["groups"],
        )


@_register_param_extract(MaxPool2dOpr, AvgPool2dOpr)
class Pooling2DExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        opr = self._opr
        param = {}
        param["mode"] = opr.mode
        param["kernel_size"] = opr.kernel_size
        param["stride"] = opr.stride
        assert (
            opr.auto_pad == "NOTSET"
        ), "ONNX To MegEngine Convert Only supports NOTSET pad mode in Pool2D"
        assert (
            opr.ceil_mode == 0
        ), "ONNX To MegEngine Convert Cannot support Ceil Mode in Pool2D"

        if opr.name == "MaxPool2d":
            assert opr.dilations == (
                1,
                1,
            ), "ONNX To MegEngine Convert Cannot support dilations in MaxPool2D"
            assert (
                opr.storage_order == 0
            ), "ONNX To MegEngine Convert Only supports row major in MaxPool2D"

        param["padding"] = convert_onnx_padding_to_mge_padding(opr.padding)

        return param


@_register_op(MaxPool2dOpr, AvgPool2dOpr)
class Pooling2DConverter(SISOConvert):
    def forward(self, inps):
        if self._opr.name == "MaxPool2d":
            return F.max_pool2d(
                inps[0],
                kernel_size=self.param["kernel_size"],
                stride=self.param["stride"],
                padding=self.param["padding"],
            )
        else:
            return F.avg_pool2d(
                inps[0],
                kernel_size=self.param["kernel_size"],
                stride=self.param["stride"],
                padding=self.param["padding"],
                mode=self.param["mode"],
            )


@_register_param_extract(MatMulOpr)
class MatMulExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        opr = self._opr
        return {
            "transA": opr.transpose_a,
            "transB": opr.transpose_b,
            "alpha": opr.alpha,
            "beta": opr.beta,
        }


@_register_op(MatMulOpr)
class MatrixMulConvert(OperatorBaseConverter):
    def forward(self, inps):
        x = F.matmul(inps[0], inps[1], self.param["transA"], self.param["transB"])
        if self.param["alpha"] != 1.0:
            x = F.mul(x, self.param["alpha"])
        if len(inps) == 3:
            if self.param["beta"] != 1.0:
                x = F.add(x, F.mul(inps[2], self.param["beta"]))
            else:
                x = F.add(x, inps[2])
        return x


@_register_param_extract(SoftmaxOpr)
class SoftmaxExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        opr = self._opr
        assert isinstance(opr.axis, int), "axis in Softmax should be int"
        return {"axis": opr.axis}


@_register_op(SoftmaxOpr)
class SoftmaxConvert(SISOConvert):
    def forward(self, inps):
        return F.nn.softmax(inps[0], self.param["axis"])


@_register_param_extract(FlattenOpr)
class FlattenExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        opr = self._opr
        assert isinstance(opr.start_axis, int), "start axis in Flatten should be int"
        assert isinstance(opr.end_axis, int), "end axis in Flatten should be int"
        return {"start_axis": opr.start_axis, "end_axis": opr.end_axis}


@_register_op(FlattenOpr)
class FlattenConvert(SISOConvert):
    def forward(self, inps):
        return F.flatten(inps[0], self.param["start_axis"], self.param["end_axis"])


@_register_param_extract(ClipOpr)
class ClipExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        opr = self._opr
        return {"upper": opr.upper, "lower": opr.lower}


@_register_op(ClipOpr)
class ClipConvert(SISOConvert):
    def forward(self, inps):
        return F.clip(inps[0], self.param["lower"], self.param["upper"])


@_register_param_extract(ConcatOpr)
class ConcatExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        opr = self._opr
        return {"axis": opr.axis}


@_register_op(ConcatOpr)
class ConcatConvert(OperatorBaseConverter):
    def forward(self, inps):
        return F.concat(inps, self.param["axis"])


@_register_param_extract(DropoutOpr)
class DropoutExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        opr = self._opr
        return {"ratio": opr.drop_prob, "training": opr.training}


@_register_op(DropoutOpr)
class DropoutConvert(SISOConvert):
    def forward(self, inps):
        return F.dropout(inps[0], self.param["ratio"], self.param["training"])


@_register_param_extract(ResizeOpr)
class ResizeExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        extra_param = self._opr.extra_param
        param = {}
        param["mode"] = extra_param["mode"]
        param["align_corners"] = None
        if "sizes" in extra_param.keys():
            param["sizes"] = tuple(
                [int(extra_param["sizes"][2]), int(extra_param["sizes"][3])]
            )
        else:
            param["scale"] = tuple(
                [float(extra_param["scale"][2]), float(extra_param["scale"][3])]
            )

        assert (
            extra_param["nearest_mode"] == "floor"
        ), "MegEngine floors as Default when Resize Mode is Nearest"
        if extra_param["coordinate_transformation_mode"] == "align_corners":
            param["align_corners"] = True
        elif extra_param["coordinate_transformation_mode"] != "asymmetric":
            raise AssertionError(
                "Mge Only supports coordinate_transformation_mode with asymmetric or align_corners"
            )

        return param


@_register_op(ResizeOpr)
class ResizeConvert(SISOConvert):
    def forward(self, inps):
        if "sizes" in self.param.keys():
            return F.vision.interpolate(
                inps[0],
                size=self.param["sizes"],
                mode=self.param["mode"],
                align_corners=self.param["align_corners"],
            )
        else:
            return F.vision.interpolate(
                inps[0],
                scale_factor=self.param["scale"],
                mode=self.param["mode"],
                align_corners=self.param["align_corners"],
            )


@_register_param_extract(LstmOpr)
class LstmExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        op = self._opr
        num_directions = 2 if op.direction == "bidirectional" else 1
        return {
            "batch_first": op.batch_first,
            "batch": op.batch_size,
            "hidden_size": op.hidden_size,
            "num_directions": num_directions,
        }


@_register_mge_module(LstmOpr)
class LstmModuleGen:
    def __init__(self, opr):
        self._opr = opr

    def mge_module_gen(self, inputs_name, outputs_name, map_op_name_2_mge_module):
        op_name = ""
        for name in inputs_name:
            op_name = op_name + name
        for name in outputs_name:
            op_name = op_name + name

        op = self._opr
        assert op.activation_alpha == None
        assert op.activation_beta == None
        assert op.activations == None
        assert op.clip == None
        assert op.input_forget == None
        assert op.output_sequence == None
        assert op.sequence_lens == None
        assert op.p == None

        if mge_version > "1.7.0":
            module = M.rnn.LSTM(
                op.input_size,
                op.hidden_size,
                op.num_layers,
                op.bias,
                op.batch_first,
                op.dropout,
                op.direction == "bidirectional",
                op.proj_size,
            )
        else:
            raise Exception("LSTM is supported begin MegEngine V1.8.0")

        parameter = list(module.named_parameters())
        assert (
            parameter[0][0] == "_flatten_weights"
        ), "First parameter of LSTM should be flatten weights"

        # fill parameters
        datas = [np.empty((4 * op.hidden_size, 0))]
        for k in range(op.num_layers):
            datas.append(op.bias_hh_l[k])
            datas.append(op.bias_ih_l[k])
            datas.append(op.weight_hh_l[k])
            datas.append(op.weight_ih_l[k])
            datas[0] = np.append(datas[0], op.weight_ih_l[k].flatten())
            datas[0] = np.append(datas[0], op.weight_hh_l[k].flatten())
            datas[0] = np.append(datas[0], op.bias_ih_l[k].flatten())
            datas[0] = np.append(datas[0], op.bias_hh_l[k].flatten())
            if op.direction == "bidirectional":
                datas.append(op.bias_hh_l_reverse[k])
                datas.append(op.bias_ih_l_reverse[k])
                datas.append(op.weight_hh_l_reverse[k])
                datas.append(op.weight_ih_l_reverse[k])
                datas[0] = np.append(datas[0], op.weight_ih_l_reverse[k].flatten())
                datas[0] = np.append(datas[0], op.weight_hh_l_reverse[k].flatten())
                datas[0] = np.append(datas[0], op.bias_ih_l_reverse[k].flatten())
                datas[0] = np.append(datas[0], op.bias_hh_l_reverse[k].flatten())

        for d, p in zip(datas, parameter):
            p[1]._reset(d)

        map_op_name_2_mge_module[op_name] = module


@_register_op(LstmOpr)
class LstmConvert(OperatorBaseConverter):
    def convert(
        self,
        inputs,
        dtypes,
        shapes,
        datas,
        outputs,
        map_ir_tensor_2_mge_tensor,
        map_op_name_2_mge_module,
    ):
        """
        Do check, find, forward and set with Mge Tensors
        """
        self.check_valid(inputs, outputs)
        inps = self.get_inputs(
            map_ir_tensor_2_mge_tensor, inputs, dtypes, shapes, datas
        )
        op_name = ""
        for name in inputs:
            op_name = op_name + name
        for name in outputs:
            op_name = op_name + name

        nr_inp = len(inps)
        if nr_inp == 1:
            x, _ = map_op_name_2_mge_module[op_name](inps[0])
        else:
            assert nr_inp == 3, "LSTM Opr's inps should be 1 or 3"
            x, _ = map_op_name_2_mge_module[op_name](inps[0], hx=[inps[1], inps[2]])

        if self.param["batch_first"]:
            target_shape = [
                self.param["batch"],
                -1,
                self.param["num_directions"],
                self.param["hidden_size"],
            ]
            pattern = [1, 2, 0, 3]
        else:
            target_shape = [
                -1,
                self.param["batch"],
                self.param["num_directions"],
                self.param["hidden_size"],
            ]
            pattern = [0, 2, 1, 3]

        x = F.reshape(x, target_shape)
        x = F.transpose(x, pattern)

        self.set_outputs(map_ir_tensor_2_mge_tensor, x, outputs)
        return x


@_register_param_extract(GatherOpr)
class GatherExtractor:
    def __init__(self, opr):
        self._opr = opr

    def extract(self):
        opr = self._opr
        return {"axis": opr.axis}


@_register_op(GatherOpr)
class GatherConvert(TISOConvert):
    def forward(self, inps):
        indices_shape = inps[1].shape.numpy()
        if inps[0].ndim == 2 and inps[1].ndim == 1:
            if self.param["axis"] == 0:
                x = F.reshape(inps[1], [indices_shape[0], 1])
            else:
                assert self.param["axis"] == 1
                x = F.reshape(inps[1], [1, indices_shape[0]])
            return F.gather(inps[0], self.param["axis"], x)
        else:
            return F.gather(inps[0], self.param["axis"], inps[1])
