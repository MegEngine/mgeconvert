# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import onnx

from ..mge_context import (
    AxisAddRemoveOpr,
    BatchNormForwardOpr,
    BroadcastOpr,
    ConcatOpr,
    ConvolutionBackwardDataOpr,
    ConvolutionBackwardFilterOpr,
    ConvolutionForwardOpr,
    DimshuffleOpr,
    ElemwiseOpr,
    GetVarShapeOpr,
    Host2DeviceCopyOpr,
    IdentityOpr,
    MarkNoBroadcastElemwiseOpr,
    MatrixMulOpr,
    MultipleDeviceTensorHolderOpr,
    PoolingForwardOpr,
    ReduceOpr,
    ReshapeOpr,
    SharedDeviceTensorOpr,
    SubtensorOpr,
    TypeCvtOpr,
    get_symvar_value,
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
        for idx, inp in enumerate(self._opr.inp_vars):
            if idx not in exclude_idx:
                if self._opr.inp_vars[idx].np_data is not None:
                    inp_tensor = onnx.helper.make_tensor_value_info(
                        inp.name, mge2onnx_dtype_mapping[inp.dtype], inp.shape
                    )
                    param = onnx.numpy_helper.from_array(inp.np_data, inp.name)
                    self._net_sources.append(inp_tensor)
                    self._parameters.append(param)

        return [var.name for var in self._opr.inp_vars]

    def _get_outputs(self):
        """
        Returns the names of outputs of onnx operator.
        """
        return [var.name for var in self._opr.out_vars]

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
                name=self._opr.name,
                **self._get_attrs(),
            )
        ]
        return nodes, self._net_sources, self._parameters


@_register_op(MultipleDeviceTensorHolderOpr, SharedDeviceTensorOpr)
class IgnoredOperatorConverter(OperatorBaseConverter):
    def convert(self):
        return [], [], []


@_register_op(GetVarShapeOpr)
class GetVarShapeConverter(OperatorBaseConverter):
    __opr_type__ = "Shape"

    def convert(self):
        shape = self._opr.out_vars[0]
        shape.np_data = get_symvar_value(shape._var).astype(np.int64)
        shape.dtype = np.int64
        return [], [], []


@_register_op(ElemwiseOpr, IdentityOpr, MarkNoBroadcastElemwiseOpr)
class ElemwiseConverter(OperatorBaseConverter):

    support_op_map = {
        "ADD": "Add",
        "SUB": "Sub",
        "TRUE_DIV": "Div",
        "MUL": "Mul",
        "RELU": "Relu",
        "FUSE_ADD_RELU": "FUSE_ADD_RELU",
        "FUSE_MUL_ADD3": "FUSE_MUL_ADD3",
        "Identity": "Identity",
        "EXP": "Exp",
        "TANH": "Tanh",
        "SIGMOID": "Sigmoid",
        "ABS": "Abs",
        "LOG": "Log",
        "FLOOR": "Floor",
        "CEIL": "Ceil",
        "POW": "Pow",
        "MAX": "Max",
    }

    def __init__(self, opr):
        super().__init__(opr)
        assert (
            opr.mode in self.support_op_map
        ), "Elemwise op doesn't support mode {}, you can implement it in ElemwiseConverter".format(
            opr.mode
        )
        op_type = self.support_op_map[opr.mode]
        self.__opr_type__ = op_type

    def convert(self):
        if self.__opr_type__ == "FUSE_ADD_RELU":
            inputs = self._get_inputs()
            outputs = self._get_outputs()
            tmp_tensor = outputs[0] + "tmp_onnx"
            add = onnx.helper.make_node("Add", inputs, [tmp_tensor])
            relu = onnx.helper.make_node("Relu", [tmp_tensor], [outputs[0]])
            return [add, relu], self._net_sources, self._parameters
        if self.__opr_type__ == "FUSE_MUL_ADD3":
            inputs = self._get_inputs()
            outputs = self._get_outputs()
            tmp_tensor = outputs[0] + "tmp_onnx"
            mul = onnx.helper.make_node("Mul", [inputs[0], inputs[1]], [tmp_tensor])
            add = onnx.helper.make_node("Add", [tmp_tensor, inputs[2]], [outputs[0]])
            return [mul, add], self._net_sources, self._parameters
        else:
            return super().convert()


@_register_op(SubtensorOpr)
class SubtensorConverter(OperatorBaseConverter):

    __opr_type__ = "Slice"

    def slice_version_1(self, starts, ends, axes, _, inputs, outputs):
        attr = {"axes": axes, "ends": ends, "starts": starts}
        slice_op = onnx.helper.make_node("Slice", inputs, outputs, **attr)
        return slice_op, [], []

    def slice_version_10(self, starts, ends, axes, steps, inputs, outputs):
        op_name = self._opr.name
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
        begin_param = np.array(opr.begin_param, dtype=np.int32)
        end_param = np.array(opr.end_param, dtype=np.int32)
        step_param = np.array(opr.step_param, dtype=np.int32)
        axis_param = np.array(opr.axis, dtype=np.int32)
        inputs = [self._get_inputs(exclude_idx=list(range(1, len(opr.inp_vars))))[0]]
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


@_register_op(DimshuffleOpr)
class DimshuffleConverter(OperatorBaseConverter):

    __opr_type__ = "Transpose"

    def _get_attrs(self):
        return {"perm": list(self._opr.pattern)}


@_register_op(MatrixMulOpr)
class MatrixMulConvert(OperatorBaseConverter):
    def convert(self):
        opr = self._opr
        const_0 = opr.name + "_const_0_onnx"
        const_0_tensor = onnx.helper.make_tensor_value_info(
            const_0, mge2onnx_dtype_mapping[np.float32], [1]
        )
        const_0_param = onnx.numpy_helper.from_array(
            np.array([0]).astype("float32"), const_0
        )
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        gemm = onnx.helper.make_node(
            "Gemm",
            [inputs[0], inputs[1], const_0],
            [outputs[0]],
            alpha=1.0,
            beta=0.0,
            transA=opr.transposeA,
            transB=opr.transposeB,
        )

        return (
            [gemm],
            self._net_sources + [const_0_tensor],
            self._parameters + [const_0_param],
        )


@_register_op(ReshapeOpr)
class ReshapeConverter(OperatorBaseConverter):

    __opr_type__ = "Reshape"

    def convert(self):
        inputs = self._get_inputs(exclude_idx=[1])
        outputs = self._get_outputs()
        inp_var = self._opr.inp_vars
        inputs[1] = self._opr.name + "shape_onnx"
        shape_tensor = onnx.helper.make_tensor_value_info(
            inputs[1], mge2onnx_dtype_mapping[np.int64], inp_var[1].shape
        )
        shape_param = onnx.numpy_helper.from_array(
            self._opr.shape_param.astype(np.int64), inputs[1]
        )
        reshape = onnx.helper.make_node("Reshape", inputs, outputs)
        return (
            [reshape],
            self._net_sources + [shape_tensor],
            self._parameters + [shape_param],
        )


@_register_op(Host2DeviceCopyOpr)
class DataProviderConverter(OperatorBaseConverter):
    def convert(self):
        out_vars = self._opr.out_vars
        inp_tensor_list = [
            onnx.helper.make_tensor_value_info(
                var.name, mge2onnx_dtype_mapping[var.dtype], var.shape
            )
            for var in out_vars
        ]

        return [], inp_tensor_list, []


@_register_op(ConvolutionForwardOpr, ConvolutionBackwardDataOpr)
class Conv2DConverter(OperatorBaseConverter):

    __opr_type__ = "Conv"

    def _get_attrs(self):
        opr = self._opr
        return {
            "kernel_shape": [opr.kh, opr.kw],
            "pads": [opr.ph, opr.pw, opr.ph, opr.pw],
            "strides": [opr.sh, opr.sw],
            "dilations": [opr.dilation_h, opr.dilation_w],
            "group": opr.group if opr.group is not None else 1,
        }

    def convert(self):
        opr = self._opr
        attrs = self._get_attrs()
        nodes = []
        exclude_idx = [0] if attrs["group"] != 1 else []
        inputs = self._get_inputs(exclude_idx)
        if isinstance(self._opr, ConvolutionBackwardDataOpr):
            inputs = [inputs[1], inputs[0]]

        outputs = self._get_outputs()
        if attrs["group"] != 1:
            w_idx = 0 if isinstance(self._opr, ConvolutionBackwardDataOpr) else 1
            flt_shape = self._opr.inp_vars[w_idx].shape
            flt_shape = [
                flt_shape[0] * flt_shape[1],
                flt_shape[2],
                flt_shape[3],
                flt_shape[4],
            ]

            if opr.param_W is not None:
                inputs[1] = opr.name + "_filter_reshape_onnx"
                flt = opr.param_W
                flt_data = flt.reshape(flt_shape)
                flt_tensor = onnx.helper.make_tensor_value_info(
                    inputs[1], mge2onnx_dtype_mapping[flt.dtype.type], flt_shape
                )
                flt_param = onnx.numpy_helper.from_array(flt_data, inputs[1])
                self._net_sources.append(flt_tensor)
                self._parameters.append(flt_param)
            else:
                reshape_inputs = [inputs[1], opr.name + "shape_onnx"]
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
                    "Reshape", reshape_inputs, [opr.name + "_filter_reshape_onnx"]
                )
                inputs[1] = opr.name + "_filter_reshape_onnx"
                nodes.append(reshape)
        onnx_op = "Conv"
        if isinstance(self._opr, ConvolutionBackwardDataOpr):
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
            np.array([1, opr.ci // opr.group, 1, 1]), grad_out_tile_in
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
            np.array([opr.no * opr.co * opr.ci // opr.group, 1, opr.ho, opr.wo]),
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
            np.array([1, opr.ni * opr.ci, opr.hi, opr.wi]), src_reshape_in
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
            kernel_shape=[opr.ho, opr.wo],
            strides=[opr.dilate_h, opr.dilate_w],
            pads=[opr.ph, opr.pw, opr.ph, opr.pw],
            dilations=[opr.sh, opr.sw],
            group=opr.ci * opr.ni,
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
            ends=[opr.kh, opr.kw],
        )
        nodes.append(grad_weight_slice)

        # Reshape
        # grad_weight: (1, no x co x ci // group, kh, kw) -> (no, co x ci // group, kh, kw)
        grad_weight_reshape_in = outputs[0] + "_grad_weight_reshape_in"
        grad_weight_reshape_source = onnx.helper.make_tensor_value_info(
            grad_weight_reshape_in, mge2onnx_dtype_mapping[np.int64], (4,)
        )
        grad_weight_reshape_param = onnx.numpy_helper.from_array(
            np.array([opr.no, opr.co * opr.ci // opr.group, opr.kh, opr.kw]),
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
            np.array([opr.ci // opr.group, opr.co, opr.kh, opr.kw]),
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


@_register_op(PoolingForwardOpr)
class Pooling2DConverter(OperatorBaseConverter):
    support_op_map = {
        "AVERAGE": "AveragePool",
        "AVERAGE_COUNT_EXCLUDE_PADDING": "AveragePool",
        "MAX": "MaxPool",
    }

    def __init__(self, opr):
        super().__init__(opr)
        assert (
            opr.mode in self.support_op_map
        ), "Pooling op doesn't support mode {}, you can implement it in Pooling2DConverter".format(
            opr.mode
        )
        self.exclude_pad = opr.mode == "AVERAGE_COUNT_EXCLUDE_PADDING"
        self.__opr_type__ = self.support_op_map[opr.mode]

    def _get_attrs(self):
        opr = self._opr
        attribute = {
            "kernel_shape": [opr.kh, opr.kw],
            "pads": [opr.ph, opr.pw, opr.ph, opr.pw],
            "strides": [opr.sh, opr.sw],
        }

        if self.__opr_type__ == "AveragePool":
            attribute["count_include_pad"] = 0 if self.exclude_pad else 1

        return attribute


@_register_op(BatchNormForwardOpr)
class BatchnormConverter(OperatorBaseConverter):
    def convert(self):
        inputs = self._get_inputs(exclude_idx=[1, 2, 3, 4])
        outputs = self._get_outputs()
        scale_ = self._opr.scale
        bias_ = self._opr.bias
        mean_ = self._opr.mean
        var_ = self._opr.var
        inputs[1] = self._opr.name + "scale_onnx"
        inputs[2] = self._opr.name + "bias_onnx"
        inputs[3] = self._opr.name + "mean_onnx"
        inputs[4] = self._opr.name + "var_onnx"
        scale = onnx.helper.make_tensor_value_info(
            inputs[1], mge2onnx_dtype_mapping[self._opr.inp_vars[1].dtype], scale_.shape
        )
        bias = onnx.helper.make_tensor_value_info(
            inputs[2], mge2onnx_dtype_mapping[self._opr.inp_vars[2].dtype], bias_.shape
        )
        mean = onnx.helper.make_tensor_value_info(
            inputs[3], mge2onnx_dtype_mapping[self._opr.inp_vars[3].dtype], mean_.shape
        )
        var = onnx.helper.make_tensor_value_info(
            inputs[4], mge2onnx_dtype_mapping[self._opr.inp_vars[4].dtype], var_.shape
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
        return {"axes": [self._opr.axis]}

    def convert(self):
        inputs = self._get_inputs()
        outputs = self._get_outputs()
        nodes = onnx.helper.make_node(
            self.__opr_type__, [inputs[0]], outputs, **self._get_attrs()
        )
        return [nodes], self._net_sources, self._parameters


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
        target_dtype = self._opr.out_vars[0].dtype
        node = onnx.helper.make_node(
            "Cast", inputs, outputs, to=mge2onnx_dtype_mapping[target_dtype]
        )
        return [node], self._net_sources, self._net_sources
