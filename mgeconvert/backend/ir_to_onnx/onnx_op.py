# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import onnx

from ...converter_ir.ir_op import (
    AbsOpr,
    AdaptiveAvgPool2dOpr,
    AddOpr,
    AvgPool2dOpr,
    AxisAddRemoveOpr,
    BatchNormalizationOpr,
    BroadcastOpr,
    CeilOpr,
    ConcatOpr,
    ConstantOpr,
    Conv2dOpr,
    ConvolutionBackwardFilterOpr,
    Deconv2dOpr,
    DropoutOpr,
    ExpOpr,
    FlattenOpr,
    FloorOpr,
    FuseMulAdd3Opr,
    GetSubTensorOpr,
    GetVarShapeOpr,
    HardSigmoidOpr,
    HardSwishOpr,
    IdentityOpr,
    LinearOpr,
    LinspaceOpr,
    LogOpr,
    MatMulOpr,
    MaxOpr,
    MaxPool2dOpr,
    MinOpr,
    MulOpr,
    MultipleDeviceTensorHolderOpr,
    PowOpr,
    ReduceOpr,
    Relu6Opr,
    ReluOpr,
    RepeatOpr,
    ReshapeOpr,
    ResizeOpr,
    SharedDeviceTensorOpr,
    SigmoidOpr,
    SiLUOpr,
    SoftmaxOpr,
    SqueezeOpr,
    SubOpr,
    TanHOpr,
    TransposeOpr,
    TrueDivOpr,
    TypeCvtOpr,
)
from ...frontend.mge_to_ir.mge_utils import get_symvar_value

mge2onnx_dtype_mapping = {
    # pylint: disable=no-member
    np.float32: onnx.TensorProto.FLOAT,
    np.float16: onnx.TensorProto.FLOAT16,
    np.int8: onnx.TensorProto.INT8,
    np.int16: onnx.TensorProto.INT16,
    np.int32: onnx.TensorProto.INT32,
    np.int64: onnx.TensorProto.INT64,
    np.uint8: onnx.TensorProto.UINT8,
}

MGE2ONNX = {}


def _register_op(*oprs):
    def callback(impl):
        for opr in oprs:
            MGE2ONNX[opr] = impl
        return impl

    return callback


opset_version = 8


def set_opset_version(version):
    global opset_version  # pylint: disable=W0603
    opset_version = version


def expand(x):
    if isinstance(x, (list, tuple)):
        return x
    elif isinstance(x, int):
        return x, x
    else:
        raise TypeError(
            "get error type! got {} expect int or tuple[int,..]".format(type(x))
        )


mge2onnx_dtype_mapping = {
    # pylint: disable=no-member
    np.float32: onnx.TensorProto.FLOAT,
    np.float16: onnx.TensorProto.FLOAT16,
    np.int8: onnx.TensorProto.INT8,
    np.int16: onnx.TensorProto.INT16,
    np.int32: onnx.TensorProto.INT32,
    np.int64: onnx.TensorProto.INT64,
    np.uint8: onnx.TensorProto.UINT8,
}


def _add_input_tensors(inputs):
    inp_tensor_list = [
        onnx.helper.make_tensor_value_info(
            tensor.name, mge2onnx_dtype_mapping[tensor.dtype], tensor.shape
        )
        for tensor in inputs
    ]
    return [], inp_tensor_list, []


class OperatorBaseConverter:

    __opr_type__ = "OperatorBaseConverter"

    def __init__(self, opr):
        """
        :param opr: the operator that converter converts.
        :type opr: subclass of :class:`.MgeOpr`
        """
        self._opr = opr
        self._net_sources = []
        self._parameters = []

    def _get_inputs(self, exclude_idx=None):
        """
        Returns the names of inputs of onnx operator.
        """
        if exclude_idx is None:
            exclude_idx = []
        for idx, inp in enumerate(self._opr.inp_tensors):
            if idx not in exclude_idx:
                if self._opr.inp_tensors[idx].np_data is not None:
                    inp_tensor = onnx.helper.make_tensor_value_info(
                        inp.name, mge2onnx_dtype_mapping[inp.dtype], inp.shape
                    )
                    param = onnx.numpy_helper.from_array(inp.np_data, inp.name)
                    self._net_sources.append(inp_tensor)
                    self._parameters.append(param)

        return [tensor.name for tensor in self._opr.inp_tensors]

    def _get_outputs(self):
        """
        Returns the names of outputs of onnx operator.
        """
        return [tensor.name for tensor in self._opr.out_tensors]

    def _get_attrs(self):
        """
        Returns extra attributes needed by :method:`.convert`
        """
        return {}

    def convert(self):
        """
        Convert owning operator to onnx operator. Could be override by
        subclass.

        Returns tuple (nodes, net_sources, parameters)
        """
        nodes = [
            onnx.helper.make_node(
                self.__opr_type__,
                self._get_inputs(),
                self._get_outputs(),
                name=self._opr.out_tensors[0].name,
                **self._get_attrs(),
            )
        ]
        return nodes, self._net_sources, self._parameters


@_register_op(
    AddOpr,
    SubOpr,
    TrueDivOpr,
    MulOpr,
    MinOpr,
    ReluOpr,
    ExpOpr,
    TanHOpr,
    SigmoidOpr,
    AbsOpr,
    LogOpr,
    FloorOpr,
    CeilOpr,
    PowOpr,
    MaxOpr,
    FuseMulAdd3Opr,
    IdentityOpr,
    SiLUOpr,
)
class ElemwiseConverter(OperatorBaseConverter):

    support_op_map = {
        AddOpr: "Add",
        SubOpr: "Sub",
        TrueDivOpr: "Div",
        MulOpr: "Mul",
        ReluOpr: "Relu",
        Relu6Opr: "FUSE_RELU6",
        ExpOpr: "Exp",
        TanHOpr: "Tanh",
        SiLUOpr: "SiLU",
        SigmoidOpr: "Sigmoid",
        AbsOpr: "Abs",
        LogOpr: "Log",
        FloorOpr: "Floor",
        CeilOpr: "Ceil",
        PowOpr: "Pow",
        MaxOpr: "Max",
        MinOpr: "Min",
        IdentityOpr: "Identity",
    }

    def __init__(self, opr):
        super().__init__(opr)
        assert isinstance(
            opr, tuple(self.support_op_map.keys())
        ), "Elemwise op doesn't support mode {}, you can implement it in ElemwiseConverter".format(
            type(opr)
        )
        op_type = self.support_op_map[type(opr)]
        self.__opr_type__ = op_type

    def convert(self):
        if self.__opr_type__ == "SiLU":
            opr = self._opr
            inputs = self._get_inputs()
            outputs = self._get_outputs()
            nodes = []
            neg_x = inputs[0] + "_negtive"
            neg_node = onnx.helper.make_node("Neg", [inputs[0]], [neg_x])
            nodes.append(neg_node)
            exp = neg_x + "_exp"
            exp_node = onnx.helper.make_node("Exp", [neg_x], [exp])
            nodes.append(exp_node)
            const_1 = inputs[0] + "_const_1"
            const_1_node = onnx.helper.make_node(
                "Constant",
                [],
                [const_1],
                value=onnx.helper.make_tensor(
                    const_1, mge2onnx_dtype_mapping[opr.inp_tensors[0].dtype], [], [1.0]
                ),
            )
            nodes.append(const_1_node)
            add = exp + "_add_const_1"
            add_node = onnx.helper.make_node("Add", [exp, const_1], [add])
            nodes.append(add_node)
            div_node = onnx.helper.make_node("Div", [inputs[0], add], outputs)
            nodes.append(div_node)
            return nodes, self._net_sources, self._parameters
        else:
            return super().convert()


@_register_op(MultipleDeviceTensorHolderOpr, SharedDeviceTensorOpr, LinspaceOpr)
class IgnoredOperatorConverter(OperatorBaseConverter):
    def convert(self):
        return [], [], []


@_register_op(GetVarShapeOpr)
class GetVarShapeConverter(OperatorBaseConverter):
    __opr_type__ = "Shape"

    def convert(self):
        shape = self._opr.out_tensors[0]
        shape.np_data = get_symvar_value(shape._var).astype(np.int64)
        shape.dtype = np.int64
        return [], [], []


@_register_op(SoftmaxOpr)
class SoftmaxConverter(OperatorBaseConverter):
    __opr_type__ = "Softmax"

    def _get_attrs(self):
        if self._opr.axis is not None:
            return {"axis": self._opr.axis}
        else:
            return {}

    def convert(self):
        opr = self._opr
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        nodes = []
        assert opr.inp_tensors[0].ndim == 2, "ONNX Softmax only support dim=2"
        offset_name = inputs[0] + "_max_offset"
        offset = onnx.helper.make_node(
            "ReduceMax",
            inputs=[inputs[0]],
            outputs=[offset_name],
            axes=[opr.axis],
            keepdims=True,
        )
        nodes.append(offset)
        sub_name = inputs[0] + "_sub_offset"
        sub = onnx.helper.make_node(
            "Sub", inputs=[inputs[0], offset_name], outputs=[sub_name],
        )
        nodes.append(sub)
        softmax = onnx.helper.make_node(
            "Softmax", inputs=[sub_name], outputs=[outputs[0]], **self._get_attrs(),
        )
        nodes.append(softmax)
        return nodes, self._net_sources, self._parameters


@_register_op(GetSubTensorOpr)
class SubtensorConverter(OperatorBaseConverter):

    __opr_type__ = "Slice"

    def slice_version_1(self, starts, ends, axes, _, inputs, outputs):
        attr = {"axes": axes, "ends": ends, "starts": starts}
        slice_op = onnx.helper.make_node("Slice", inputs, outputs, **attr)
        return slice_op, [], []

    def slice_version_10(self, starts, ends, axes, steps, inputs, outputs):
        op_name = self._opr.out_tensors[0].name
        inputs = inputs + [
            op_name + "_begin",
            op_name + "_end",
            op_name + "_axis",
            op_name + "_step",
        ]
        begin = onnx.helper.make_tensor_value_info(
            inputs[1], mge2onnx_dtype_mapping[np.int32], starts.shape
        )
        end = onnx.helper.make_tensor_value_info(
            inputs[2], mge2onnx_dtype_mapping[np.int32], ends.shape
        )
        axis = onnx.helper.make_tensor_value_info(
            inputs[3], mge2onnx_dtype_mapping[np.int32], axes.shape
        )
        step = onnx.helper.make_tensor_value_info(
            inputs[4], mge2onnx_dtype_mapping[np.int32], steps.shape
        )
        net_sources = [begin, end, axis, step]
        parameters = [
            onnx.numpy_helper.from_array(starts, inputs[1]),
            onnx.numpy_helper.from_array(ends, inputs[2]),
            onnx.numpy_helper.from_array(axes, inputs[3]),
            onnx.numpy_helper.from_array(steps, inputs[4]),
        ]
        Slice = onnx.helper.make_node("Slice", inputs, outputs)
        return Slice, net_sources, parameters

    def convert(self):
        opr = self._opr
        squeeze_axis = opr.squeeze_axis
        begin_param = np.array(opr.begin_params, dtype=np.int32)
        end_param = np.array(opr.end_params, dtype=np.int32)
        step_param = np.array(opr.step_params, dtype=np.int32)
        axis_param = np.array(opr.axis, dtype=np.int32)

        inputs = [self._get_inputs(exclude_idx=list(range(1, len(opr.inp_tensors))))[0]]
        outputs = self._get_outputs()
        slice_outputs = [
            outputs[0] if len(squeeze_axis) == 0 else outputs[0] + "tmp@onnx"
        ]
        slice_op = None
        slice_net_sources = []
        slice_param = []
        if opset_version < 10:
            slice_op, slice_net_sources, slice_param = self.slice_version_1(
                begin_param, end_param, axis_param, step_param, inputs, slice_outputs
            )
        else:
            slice_op, slice_net_sources, slice_param = self.slice_version_10(
                begin_param, end_param, axis_param, step_param, inputs, slice_outputs
            )
        nodes = []
        self._parameters.extend(slice_param)
        self._net_sources.extend(slice_net_sources)
        nodes.append(slice_op)
        if len(squeeze_axis) > 0:
            Squeeze = onnx.helper.make_node(
                "Squeeze", slice_outputs, outputs, axes=squeeze_axis
            )
            nodes.append(Squeeze)

        return (nodes, self._net_sources, self._parameters)


@_register_op(TransposeOpr)
class DimshuffleConverter(OperatorBaseConverter):

    __opr_type__ = "Transpose"

    def _get_attrs(self):
        return {"perm": list(self._opr.pattern)}


@_register_op(MatMulOpr, LinearOpr)
class MatrixMulConvert(OperatorBaseConverter):
    def convert(self):
        opr = self._opr
        nodes = []
        const_0 = opr.out_tensors[0].name + "_const_0_onnx"
        const_0_tensor = onnx.helper.make_tensor_value_info(
            const_0, mge2onnx_dtype_mapping[np.float32], [1]
        )
        const_0_param = onnx.numpy_helper.from_array(
            np.array([0]).astype("float32"), const_0
        )
        self._net_sources.append(const_0_tensor)
        self._parameters.append(const_0_param)
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        if isinstance(opr, LinearOpr) and opr.has_bias:
            temp_out = inputs[0] + "_mul" + inputs[1]
            gemm = onnx.helper.make_node(
                "Gemm",
                [inputs[0], inputs[1], const_0],
                [temp_out],
                alpha=1.0,
                beta=0.0,
                transA=opr.transpose_a,
                transB=opr.transpose_b,
            )
            nodes.append(gemm)
            add_bias = onnx.helper.make_node(
                "Add", inputs=[temp_out, inputs[2]], outputs=[outputs[0]],
            )
            nodes.append(add_bias)
        else:
            gemm = onnx.helper.make_node(
                "Gemm",
                [inputs[0], inputs[1], const_0],
                [outputs[0]],
                alpha=1.0,
                beta=0.0,
                transA=opr.transpose_a,
                transB=opr.transpose_b,
            )
            nodes.append(gemm)

        return (
            nodes,
            self._net_sources,
            self._parameters,
        )


@_register_op(ReshapeOpr)
class ReshapeConverter(OperatorBaseConverter):

    __opr_type__ = "Reshape"

    def convert(self):
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        shape_tensor_name = self._opr.out_tensors[0].name + "_shape_onnx"
        shape_tensor = onnx.helper.make_tensor_value_info(
            shape_tensor_name,
            mge2onnx_dtype_mapping[np.int64],
            (len(self._opr.out_shape),),
        )
        shape_param = onnx.numpy_helper.from_array(
            np.array(self._opr.out_shape, dtype=np.int64), shape_tensor_name
        )
        reshape = onnx.helper.make_node(
            "Reshape", [inputs[0], shape_tensor_name], outputs
        )
        return (
            [reshape],
            self._net_sources + [shape_tensor],
            self._parameters + [shape_param],
        )


@_register_op(AxisAddRemoveOpr)
class AxisAddRemoveConverter(OperatorBaseConverter):
    def convert(self):
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        add_axis = []
        remove_axis = []
        for desc in self._opr.desc:
            if desc["method"] == 0:
                add_axis.append(desc["axisnum"])
            else:
                remove_axis.append(desc["axisnum"])

        if len(add_axis) > 0 and len(remove_axis) > 0:
            assert (
                False
            ), "AsixAddRemove converter doesn't support add and remove axises concurrently"

        elif len(add_axis) > 0:
            unsqueeze = onnx.helper.make_node(
                "Unsqueeze", inputs, outputs, axes=add_axis
            )
            ret = [unsqueeze]
        elif len(remove_axis) > 0:
            squeeze = onnx.helper.make_node(
                "Squeeze", inputs, outputs, axes=remove_axis
            )
            ret = [squeeze]
        else:
            ret = []
        return ret, self._net_sources, self._parameters


@_register_op(Conv2dOpr, Deconv2dOpr)
class Conv2DConverter(OperatorBaseConverter):

    __opr_type__ = "Conv"

    def _get_attrs(self):
        opr = self._opr
        ph, pw = expand(opr.padding)
        sh, sw = expand(opr.stride)
        param_W = opr.inp_tensors[1].shape
        kh, kw = param_W[-2:]
        group = opr.groups if opr.groups is not None else 1
        dilation_h, dilation_w = expand(opr.dilation)
        return {
            "kernel_shape": [kh, kw],
            "pads": [ph, pw, ph, pw],
            "strides": [sh, sw],
            "dilations": [dilation_h, dilation_w],
            "group": group if group is not None else 1,
        }

    def convert(self):
        opr = self._opr
        attrs = self._get_attrs()
        nodes = []
        exclude_idx = [0] if attrs["group"] != 1 else []
        inputs = self._get_inputs(exclude_idx)
        outputs = self._get_outputs()
        if attrs["group"] != 1:
            flt_shape = self._opr.inp_tensors[1].shape
            flt_shape = [
                flt_shape[0] * flt_shape[1],
                flt_shape[2],
                flt_shape[3],
                flt_shape[4],
            ]

            if opr.inp_tensors[1].np_data is not None:
                inputs[1] = opr.out_tensors[0].name + "_filter_reshape_onnx"
                flt = opr.inp_tensors[1].np_data
                flt_data = flt.reshape(flt_shape)
                flt_tensor = onnx.helper.make_tensor_value_info(
                    inputs[1], mge2onnx_dtype_mapping[flt.dtype.type], flt_shape
                )
                flt_param = onnx.numpy_helper.from_array(flt_data, inputs[1])
                self._net_sources.append(flt_tensor)
                self._parameters.append(flt_param)
            else:
                reshape_inputs = [inputs[1], opr.out_tensors[0].name + "shape_onnx"]
                shape_tensor = onnx.helper.make_tensor_value_info(
                    reshape_inputs[1],
                    mge2onnx_dtype_mapping[np.int64],
                    (len(flt_shape),),
                )
                shape_param = onnx.numpy_helper.from_array(
                    np.array(flt_shape, dtype="int64"), reshape_inputs[1]
                )
                self._net_sources.append(shape_tensor)
                self._parameters.append(shape_param)
                reshape = onnx.helper.make_node(
                    "Reshape",
                    reshape_inputs,
                    [opr.out_tensors[0].name + "_filter_reshape_onnx"],
                )
                inputs[1] = opr.out_tensors[0].name + "_filter_reshape_onnx"
                nodes.append(reshape)
        onnx_op = "Conv"
        if isinstance(self._opr, Deconv2dOpr):
            onnx_op = "ConvTranspose"
        conv2d = onnx.helper.make_node(onnx_op, inputs, [outputs[0]], **attrs)
        nodes.append(conv2d)
        return (nodes, self._net_sources, self._parameters)


@_register_op(ConvolutionBackwardFilterOpr)
class Conv2DBackwardFilterConverter(OperatorBaseConverter):
    def convert(self):
        opr = self._opr
        # src, grad_out, weight = self._opr.inp_vars

        nodes = []
        inputs = self._get_inputs()
        outputs = self._get_outputs()

        # Tile
        # grad_out: (no, co, ho, wo) -> (no, co x ci / group, ho, wo)
        grad_out_tile_in = outputs[0] + "_grad_out_tile_in"
        grad_out_tile_source = onnx.helper.make_tensor_value_info(
            grad_out_tile_in, mge2onnx_dtype_mapping[np.int64], (4,)
        )
        grad_out_tile_param = onnx.numpy_helper.from_array(
            np.array([1, opr.src_shape[1] // opr.group, 1, 1]), grad_out_tile_in
        )
        self._net_sources.append(grad_out_tile_source)
        self._parameters.append(grad_out_tile_param)
        grad_out_tile_out = outputs[0] + "_grad_out_tile_out"
        grad_out_tile = onnx.helper.make_node(
            "Tile", [inputs[1], grad_out_tile_in], [grad_out_tile_out]
        )
        nodes.append(grad_out_tile)

        # Reshape
        # grad_out: (no, co x ci / group, ho, wo) -> (no x co x ci / group, 1, ho, wo)
        grad_out_reshape_in = outputs[0] + "_grad_out_reshape_in"
        grad_out_reshape_source = onnx.helper.make_tensor_value_info(
            grad_out_reshape_in, mge2onnx_dtype_mapping[np.int64], (4,)
        )
        grad_out_reshape_param = onnx.numpy_helper.from_array(
            np.array(
                [
                    opr.grad_out_shape[0]
                    * opr.grad_out_shape[1]
                    * opr.src_shape[1]
                    // opr.group,
                    1,
                    opr.grad_out_shape[2],
                    opr.grad_out_shape[3],
                ]
            ),
            grad_out_reshape_in,
        )
        self._net_sources.append(grad_out_reshape_source)
        self._parameters.append(grad_out_reshape_param)
        grad_out_reshape_out = outputs[0] + "_grad_out_reshape_out"
        reshape = onnx.helper.make_node(
            "Reshape", [grad_out_tile_out, grad_out_reshape_in], [grad_out_reshape_out]
        )
        nodes.append(reshape)

        # Reshape
        # src: (ni, ci, hi, wi) -> (1, ni x ci, hi, wi)
        src_reshape_in = outputs[0] + "_src_reshape_in"
        src_reshape_source = onnx.helper.make_tensor_value_info(
            src_reshape_in, mge2onnx_dtype_mapping[np.int64], (4,)
        )
        src_reshape_param = onnx.numpy_helper.from_array(
            np.array(
                [
                    1,
                    opr.src_shape[0] * opr.src_shape[1],
                    opr.src_shape[2],
                    opr.src_shape[3],
                ]
            ),
            src_reshape_in,
        )
        self._net_sources.append(src_reshape_source)
        self._parameters.append(src_reshape_param)
        src_reshape_out = outputs[0] + "_src_reshape_out"
        reshape = onnx.helper.make_node(
            "Reshape", [inputs[0], src_reshape_in], [src_reshape_out]
        )
        nodes.append(reshape)

        # Conv:
        # group = ni * ci
        # src(1, ni x ci, hi, wi) + grad_out(no x co x ci / group, 1, ho, wo)
        # -> grad_weight(1, no x co x ci / group, ?, ?)
        grad_weight = outputs[0] + "_grad_weight"
        grad_weight_conv = onnx.helper.make_node(
            "Conv",
            [src_reshape_out, grad_out_reshape_out],
            [grad_weight],
            kernel_shape=[opr.grad_out_shape[2], opr.grad_out_shape[3]],
            strides=[opr.dilation[0], opr.dilation[1]],
            pads=[opr.padding[0], opr.padding[1], opr.padding[0], opr.padding[1]],
            dilations=[opr.stride[0], opr.stride[1]],
            group=opr.src_shape[1] * opr.src_shape[0],
        )
        nodes.append(grad_weight_conv)

        # Slice
        # grad_weight: (1, no x co x ci // group, ?, ?) -> (1, no x co x ci // group, kh, kw)
        grad_weight_slice_out = outputs[0] + "_grad_weight_slice_out"
        grad_weight_slice = onnx.helper.make_node(
            "Slice",
            [grad_weight],
            [grad_weight_slice_out],
            axes=[2, 3],
            starts=[0, 0],
            ends=[opr.kernel_shape[0], opr.kernel_shape[1]],
        )
        nodes.append(grad_weight_slice)

        # Reshape
        # grad_weight: (1, no x co x ci // group, kh, kw) -> (no, co x ci // group, kh, kw)
        grad_weight_reshape_in = outputs[0] + "_grad_weight_reshape_in"
        grad_weight_reshape_source = onnx.helper.make_tensor_value_info(
            grad_weight_reshape_in, mge2onnx_dtype_mapping[np.int64], (4,)
        )
        grad_weight_reshape_param = onnx.numpy_helper.from_array(
            np.array(
                [
                    opr.grad_out_shape[0],
                    opr.grad_out_shape[1] * opr.src_shape[1] // opr.group,
                    opr.kernel_shape[0],
                    opr.kernel_shape[1],
                ]
            ),
            grad_weight_reshape_in,
        )
        self._net_sources.append(grad_weight_reshape_source)
        self._parameters.append(grad_weight_reshape_param)
        grad_weight_reshape_out = outputs[0] + "_grad_weight_reshape_out"
        reshape = onnx.helper.make_node(
            "Reshape",
            [grad_weight_slice_out, grad_weight_reshape_in],
            [grad_weight_reshape_out],
        )
        nodes.append(reshape)

        # ReduceSum
        # grad_weight: (no, co x ci // group, kh, kw) -> (1, co x ci // goup, kh, kw)
        grad_weight_reduce_out = outputs[0] + "_grad_weight_reduce_out"
        grad_weight_reduce = onnx.helper.make_node(
            "ReduceSum", [grad_weight_reshape_out], [grad_weight_reduce_out], axes=[0],
        )
        nodes.append(grad_weight_reduce)

        # Reshape
        # grad_weight: (1, co x ci // group, kh, kw) -> (ci // group, co, kh, kw)
        grad_weight_reshape2_in = outputs[0] + "_grad_weight_reshape2_in"
        grad_weight_reshape2_source = onnx.helper.make_tensor_value_info(
            grad_weight_reshape2_in, mge2onnx_dtype_mapping[np.int64], (4,)
        )
        grad_weight_reshape2_param = onnx.numpy_helper.from_array(
            np.array(
                [
                    opr.src_shape[1] // opr.group,
                    opr.grad_out_shape[1],
                    opr.kernel_shape[0],
                    opr.kernel_shape[1],
                ]
            ),
            grad_weight_reshape2_in,
        )
        self._net_sources.append(grad_weight_reshape2_source)
        self._parameters.append(grad_weight_reshape2_param)
        grad_weight_reshape2_out = outputs[0] + "_grad_weight_reshape2_out"
        reshape = onnx.helper.make_node(
            "Reshape",
            [grad_weight_reduce_out, grad_weight_reshape2_in],
            [grad_weight_reshape2_out],
        )
        nodes.append(reshape)

        # Transpose
        grad_weight_transpose_out = outputs[0]
        transpose = onnx.helper.make_node(
            "Transpose",
            [grad_weight_reshape2_out],
            [grad_weight_transpose_out],
            perm=[1, 0, 2, 3],
        )
        nodes.append(transpose)

        return (nodes, self._net_sources, self._parameters)


@_register_op(MaxPool2dOpr, AvgPool2dOpr)
class Pooling2DConverter(OperatorBaseConverter):
    support_op_map = {
        "AVERAGE": "AveragePool",
        "AVERAGE_COUNT_EXCLUDE_PADDING": "AveragePool",
        "MAX": "MaxPool",
    }

    def __init__(self, opr):
        super().__init__(opr)
        assert (
            opr.mode.upper() in self.support_op_map
        ), "Pooling op doesn't support mode {}, you can implement it in Pooling2DConverter".format(
            type(opr)
        )
        self.exclude_pad = opr.mode.upper() == "AVERAGE_COUNT_EXCLUDE_PADDING"
        self.__opr_type__ = self.support_op_map[opr.mode.upper()]

    def _get_attrs(self):
        opr = self._opr
        kh, kw = expand(opr.kernel_size)
        ph, pw = expand(opr.padding)
        sh, sw = expand(opr.stride)
        attribute = {
            "kernel_shape": [kh, kw],
            "pads": [ph, pw, ph, pw],
            "strides": [sh, sw],
        }

        if self.__opr_type__ == "AveragePool":
            attribute["count_include_pad"] = 0 if self.exclude_pad else 1

        return attribute


@_register_op(BatchNormalizationOpr)
class BatchnormConverter(OperatorBaseConverter):
    def convert(self):
        opr = self._opr
        inputs = self._get_inputs(exclude_idx=[1, 2, 3, 4])
        outputs = self._get_outputs()
        scale_ = opr.inp_tensors[1].np_data.squeeze()
        bias_ = opr.inp_tensors[2].np_data.squeeze()
        mean_ = opr.inp_tensors[3].np_data.squeeze()
        var_ = opr.inp_tensors[4].np_data.squeeze()

        inputs[1] = self._opr.inp_tensors[0].name + "_scale_onnx"
        inputs[2] = self._opr.inp_tensors[0].name + "_bias_onnx"
        inputs[3] = self._opr.inp_tensors[0].name + "_mean_onnx"
        inputs[4] = self._opr.inp_tensors[0].name + "_var_onnx"
        scale = onnx.helper.make_tensor_value_info(
            inputs[1],
            mge2onnx_dtype_mapping[self._opr.inp_tensors[1].dtype],
            scale_.shape,
        )
        bias = onnx.helper.make_tensor_value_info(
            inputs[2],
            mge2onnx_dtype_mapping[self._opr.inp_tensors[2].dtype],
            bias_.shape,
        )
        mean = onnx.helper.make_tensor_value_info(
            inputs[3],
            mge2onnx_dtype_mapping[self._opr.inp_tensors[3].dtype],
            mean_.shape,
        )
        var = onnx.helper.make_tensor_value_info(
            inputs[4],
            mge2onnx_dtype_mapping[self._opr.inp_tensors[4].dtype],
            var_.shape,
        )
        self._parameters.extend(
            [
                onnx.numpy_helper.from_array(scale_, inputs[1]),
                onnx.numpy_helper.from_array(bias_, inputs[2]),
                onnx.numpy_helper.from_array(mean_, inputs[3]),
                onnx.numpy_helper.from_array(var_, inputs[4]),
            ]
        )
        bn = onnx.helper.make_node(
            "BatchNormalization", inputs, [outputs[self._opr.output_idx]]
        )
        return ([bn], self._net_sources + [scale, bias, mean, var], self._parameters)


@_register_op(ConcatOpr)
class ConcatConverter(OperatorBaseConverter):
    __opr_type__ = "Concat"

    def __init__(self, opr):
        super().__init__(opr)
        if opset_version < 11:
            assert (
                self._opr.axis >= 0
            ), "opset {} doesn't support negative aixs in concat opr".format(
                opset_version
            )

    def _get_attrs(self):
        return {"axis": self._opr.axis}


@_register_op(ReduceOpr)
class ReduceConverter(OperatorBaseConverter):
    support_op_map = {
        "MAX": "ReduceMax",
        "SUM": "ReduceSum",
        "SUM_SQR": "ReduceSumSquare",
    }

    def __init__(self, opr):
        super().__init__(opr)
        assert (
            opr.mode in self.support_op_map
        ), "Reduce op doesn't support mode {}, you can implement it in ReduceConverter".format(
            opr.mode
        )
        self.__opr_type__ = self.support_op_map[opr.mode]

    def _get_attrs(self):
        if self._opr.axis < 2000000000:
            return {"axes": [self._opr.axis], "keepdims": self._opr.keep_dims}
        else:
            return {"axes": [0], "keepdims": self._opr.keep_dims}

    def convert(self):
        if self._opr.inp_tensors[0].shape == self._opr.out_tensors[0].shape:
            inputs = self._get_inputs()
            outputs = self._get_outputs()
            nodes = onnx.helper.make_node(
                self.__opr_type__, [inputs[0]], outputs, **self._get_attrs()
            )
            return [nodes], self._net_sources, self._parameters
        else:
            inputs = self._get_inputs()
            outputs = self._get_outputs()
            if len(inputs) > 1:
                temp_node = inputs[0] + "_reshape_in"
            else:
                temp_node = outputs[0]
            out_nodes = []
            nodes = onnx.helper.make_node(
                self.__opr_type__, [inputs[0]], [temp_node], **self._get_attrs()
            )
            out_nodes.append(nodes)
            if len(inputs) > 1:
                shape = inputs[1] + "_shape"
                shape_tensor = onnx.helper.make_tensor_value_info(
                    shape,
                    mge2onnx_dtype_mapping[np.int64],
                    self._opr.inp_tensors[1].shape,
                )
                shape_param = onnx.numpy_helper.from_array(
                    self._opr.inp_tensors[1].np_data.astype(np.int64), shape
                )
                self._net_sources.append(shape_tensor)
                self._parameters.append(shape_param)
                reshape_node = onnx.helper.make_node(
                    "Reshape", [temp_node, shape], outputs,
                )
                out_nodes.append(reshape_node)
            return out_nodes, self._net_sources, self._parameters


@_register_op(SqueezeOpr)
class SqueezeConverter(OperatorBaseConverter):
    def convert(self):
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        remove_axis = []
        for axis in self._opr.squeeze_dims:
            remove_axis.append(axis)

        if len(remove_axis) > 0:
            squeeze = onnx.helper.make_node(
                "Squeeze", inputs, outputs, axes=remove_axis
            )
            ret = [squeeze]
        else:
            ret = []
        return ret, self._net_sources, self._parameters


@_register_op(BroadcastOpr)
class BroadcastOprConverter(OperatorBaseConverter):
    def convert(self):
        assert opset_version > 7, "onnx support Expand (broadcast) since opset 8"
        inputs = self._get_inputs()
        typecvt_node = onnx.helper.make_node(
            "Cast",
            [inputs[1]],
            [inputs[1] + "_int64"],
            to=mge2onnx_dtype_mapping[np.int64],
        )
        inputs[1] = inputs[1] + "_int64"
        outputs = self._get_outputs()
        broadcast_node = onnx.helper.make_node("Expand", inputs, outputs)
        return [typecvt_node, broadcast_node], self._net_sources, self._parameters


@_register_op(TypeCvtOpr)
class TypeCvtOprConverter(OperatorBaseConverter):
    def convert(self):
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        target_dtype = self._opr.out_tensors[0].dtype
        node = onnx.helper.make_node(
            "Cast", inputs, outputs, to=mge2onnx_dtype_mapping[target_dtype]
        )
        return [node], self._net_sources, self._net_sources


@_register_op(ConstantOpr)
class ConstantOprConverter(OperatorBaseConverter):
    def convert(self):
        return [], self._net_sources, self._parameters


@_register_op(DropoutOpr)
class DropoutOprConverter(OperatorBaseConverter):
    __opr_type__ = "Dropout"


@_register_op(FlattenOpr)
class FlattenOprConverter(OperatorBaseConverter):
    def convert(self):
        opr = self._opr
        assert opr.end_axis == -1, "Onnx only support end_axis = -1"
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        nodes = []
        inp_shape = list(opr.inp_tensors[0].shape)
        if inp_shape[opr.start_axis] != 1:
            flatten = onnx.helper.make_node(
                "Flatten", inputs=inputs, outputs=outputs, axis=opr.start_axis,
            )
            nodes.append(flatten)
        else:
            tmp_name = inputs[0] + "_tmp_flatten"
            flatten = onnx.helper.make_node(
                "Flatten", inputs=inputs, outputs=[tmp_name], axis=opr.start_axis,
            )
            nodes.append(flatten)
            squeeze = onnx.helper.make_node(
                "Squeeze", [tmp_name], outputs, axes=[opr.start_axis]
            )
            nodes.append(squeeze)
        return nodes, self._net_sources, self._parameters


@_register_op(AdaptiveAvgPool2dOpr)
class AdaptiveAvgPool2dOprConverter(OperatorBaseConverter):
    __opr_type__ = "AveragePool"

    def _get_attrs(self):
        opr = self._opr
        oh, ow = expand(opr.out_shape)
        ih, iw = list(opr.inp_tensors[0].shape)[-2:]

        ph, pw = 0, 0
        sh, sw = ih // oh, iw // ow
        kh, kw = ih - (oh - 1) * sh, iw - (ow - 1) * sw

        attribute = {
            "kernel_shape": [kh, kw],
            "pads": [ph, pw, ph, pw],
            "strides": [sh, sw],
            "count_include_pad": 1,
        }

        return attribute


def gen_relu6_node(opr, inputs, outputs):
    nodes = []
    if opset_version >= 11:
        zero_name = opr.out_tensors[0].name + "_const_zero"
        zero = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[zero_name],
            value=onnx.helper.make_tensor(
                name=zero_name,
                data_type=onnx.TensorProto.FLOAT,  # pylint: disable=no-member
                dims=[],
                vals=[0.0],
            ),
        )
        six_name = opr.out_tensors[0].name + "_const_six"
        six = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[six_name],
            value=onnx.helper.make_tensor(
                name=six_name,
                data_type=onnx.TensorProto.FLOAT,  # pylint: disable=no-member
                dims=[],
                vals=[6.0],
            ),
        )
        relu6 = onnx.helper.make_node(
            "Clip", inputs=[inputs[0], zero_name, six_name], outputs=outputs,
        )
        nodes.extend([zero, six, relu6])
    else:
        relu6 = onnx.helper.make_node(
            "Clip", inputs=inputs, outputs=outputs, min=0.0, max=6.0,
        )
        nodes.append(relu6)

    return nodes


def relu6_add3_dev6(opr, inputs, outputs):
    nodes = []
    three_name = opr.out_tensors[0].name + "_const_three"
    three = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=[three_name],
        value=onnx.helper.make_tensor(
            name=three_name,
            data_type=mge2onnx_dtype_mapping[opr.inp_tensors[0].dtype],
            dims=[],
            vals=[3.0],
        ),
    )
    nodes.append(three)
    # x + 3
    add_three = opr.inp_tensors[0].name + "_add_three"
    add_three_node = onnx.helper.make_node(
        "Add", inputs=[inputs[0], three_name], outputs=[add_three]
    )
    nodes.append(add_three_node)
    # relu6(x+3)
    relu6_out = add_three + "_relu6out"
    relu6_nodes = gen_relu6_node(opr, [add_three], [relu6_out])
    nodes.extend(relu6_nodes)
    # relu6(x+3)/6
    relu6_six = relu6_out + "_six"
    six_name = opr.out_tensors[0].name + "_const_six"
    six = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=[relu6_six],
        value=onnx.helper.make_tensor(
            name=six_name,
            data_type=mge2onnx_dtype_mapping[opr.inp_tensors[0].dtype],
            dims=[],
            vals=[6.0],
        ),
    )
    nodes.append(six)
    relu6_six_dev = outputs[0]
    dev6 = onnx.helper.make_node(
        "Div", inputs=[relu6_out, relu6_six], outputs=[relu6_six_dev],
    )
    nodes.append(dev6)
    return nodes


@_register_op(Relu6Opr)
class Relu6OprConverter(OperatorBaseConverter):
    def convert(self):
        opr = self._opr
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        nodes = gen_relu6_node(opr, inputs, outputs)
        return nodes, self._net_sources, self._parameters


@_register_op(HardSwishOpr)
class HardSwishOprConverter(OperatorBaseConverter):
    def convert(self):
        opr = self._opr
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        nodes = []
        relu6_six_dev = opr.inp_tensors[0].name + "_add_three_relu6out_dev6"
        temp_nodes = relu6_add3_dev6(opr, inputs, [relu6_six_dev])
        nodes.extend(temp_nodes)
        relu6_mul = onnx.helper.make_node(
            "Mul", inputs=[inputs[0], relu6_six_dev], outputs=outputs,
        )
        nodes.append(relu6_mul)
        return nodes, self._net_sources, self._parameters


@_register_op(HardSigmoidOpr)
class HardSigmoidOprConverter(OperatorBaseConverter):
    __opr_type__ = "HardSigmoid"

    def _get_attrs(self):
        return {
            "alpha": 1.0 / 6.0,
            "beta": 0.5,
        }

    def convert(self):
        nodes = [
            onnx.helper.make_node(
                "HardSigmoid",
                self._get_inputs()[:1],
                self._get_outputs(),
                name=self._opr.out_tensors[0].name,
                **self._get_attrs(),
            )
        ]
        return nodes, self._net_sources, self._parameters


@_register_op(RepeatOpr)
class RepeatConverter(OperatorBaseConverter):
    def convert(self):
        opr = self._opr
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        nodes = []

        unsqueeze_out = inputs[0] + "_unsqueeze_out"
        unsqueeze = onnx.helper.make_node(
            "Unsqueeze",
            inputs=[inputs[0]],
            outputs=[unsqueeze_out],
            axes=[opr.axis + 1],
        )
        nodes.append(unsqueeze)

        repeat_shape = [1] * (opr.inp_tensors[0].ndim + 1)
        repeat_shape[opr.axis + 1] *= opr.repeats
        tile_repeats = unsqueeze_out + "_repeats"
        tile_repeats_tensor = onnx.helper.make_tensor_value_info(
            tile_repeats,
            mge2onnx_dtype_mapping[np.int64],
            (opr.inp_tensors[0].ndim + 1,),
        )
        tile_repeats_param = onnx.numpy_helper.from_array(
            np.array(repeat_shape).astype("int64"), tile_repeats
        )
        self._net_sources.append(tile_repeats_tensor)
        self._parameters.append(tile_repeats_param)

        repeat_name = inputs[0] + "_tile"
        repeat = onnx.helper.make_node(
            "Tile", inputs=[unsqueeze_out, tile_repeats], outputs=[repeat_name],
        )
        nodes.append(repeat)
        shape_tensor_name_after = repeat_name + "_reshape_after"
        shape_tensor_after = onnx.helper.make_tensor_value_info(
            shape_tensor_name_after,
            mge2onnx_dtype_mapping[np.int64],
            (opr.out_tensors[0].ndim,),
        )
        shape_param_after = onnx.numpy_helper.from_array(
            np.array(opr.out_tensors[0].shape, dtype=np.int64), shape_tensor_name_after
        )
        self._net_sources.append(shape_tensor_after)
        self._parameters.append(shape_param_after)
        reshape_out = onnx.helper.make_node(
            "Reshape", [repeat_name, shape_tensor_name_after], outputs
        )
        nodes.append(reshape_out)
        return nodes, self._net_sources, self._parameters


@_register_op(ResizeOpr)
class ResizeConverter(OperatorBaseConverter):
    def convert(self):
        opr = self._opr
        assert opr.mode == "nearest", "Resize mode should be NEAREST."
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        nodes = []
        _, _, h, w = opr.inp_tensors[0].shape
        if opr.out_size is not None:
            s_h, s_w = opr.out_size
            s_h = s_h / h
            s_w = s_w / w
        else:
            if isinstance(opr.scale_factor, tuple):
                s_h, s_w = opr.scale_factor
            else:
                s_h, s_w = opr.scale_factor, opr.scale_factor
        s = [1.0, 1.0, s_h, s_w]

        scales = inputs[0] + "_scales"
        scales_tensor = onnx.helper.make_tensor_value_info(
            scales, mge2onnx_dtype_mapping[np.float32], (4,)
        )
        scales_param = onnx.numpy_helper.from_array(
            np.array(s, dtype=np.float32), scales
        )

        self._net_sources.append(scales_tensor)
        self._parameters.append(scales_param)

        resize = onnx.helper.make_node(
            "Resize", inputs=[inputs[0], scales], outputs=outputs, mode="nearest",
        )
        nodes.append(resize)
        return nodes, self._net_sources, self._parameters
