# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module, super-init-not-called, non-parent-init-called

import megengine.module as M
import megengine.module.qat as QAT
import numpy as np
from megengine.tensor import Parameter
from megengine.traced_module.expr import CallMethod

from ....converter_ir.ir_op import Conv2dOpr, ConvRelu2dOpr
from ....converter_ir.ir_tensor import AxisOrder
from ....converter_ir.ir_transform import fold_conv_bn
from ..tm_utils import _unexpand, get_logger
from .base import OpGenBase, _register_op
from .conv2d import GenConvBase

logger = get_logger(__name__)


class GenConvBnBase(GenConvBase):
    def __init__(self, expr, irgraph, op_cls):
        OpGenBase.__init__(self, expr, irgraph)
        assert isinstance(expr, CallMethod)
        conv_module = expr.inputs[0].owner.conv
        self.weight = conv_module.weight
        self.bias = conv_module.bias
        self.stride = _unexpand(conv_module.stride)
        self.padding = _unexpand(conv_module.padding)
        self.dilation = _unexpand(conv_module.dilation)
        self.groups = conv_module.groups

        bn_module = expr.inputs[0].owner.bn
        self.running_mean = bn_module.running_mean
        self.running_var = bn_module.running_var
        self.bn_weight = bn_module.weight
        self.bn_bias = bn_module.bias
        self.op = op_cls(self.stride, self.padding, self.dilation, self.groups)

        if self.bias is None:
            weight_shape = self.weight.shape
            bias_shape = (
                weight_shape[0]
                if len(weight_shape) == 4
                else weight_shape[0] * weight_shape[1]
            )
            bias_shape = (1, bias_shape, 1, 1)
            self.bias = Parameter(np.zeros(bias_shape, dtype=np.float32))
        self.weight, self.bias = fold_conv_bn(
            self.weight,
            self.bias,
            self.groups,
            self.bn_weight,
            self.bn_bias,
            self.running_mean,
            self.running_var,
            bn_module.eps,
        )


@_register_op(M.ConvBn2d)
class GenConvBn2dOpr(GenConvBnBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph, Conv2dOpr)
        self.add_opr_vars(AxisOrder.OIHW)


@_register_op(M.ConvBnRelu2d)
class GenConvBnRelu2dOpr(GenConvBnBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph, ConvRelu2dOpr)
        self.add_opr_vars(AxisOrder.OIHW)


class GenQConvBnBase(GenConvBnBase):
    def __init__(self, expr, irgraph, op_cls):
        conv_module = expr.inputs[0].owner
        if hasattr(conv_module.weight_fake_quant, "get_qparams"):
            self.weight_qparams = conv_module.weight_fake_quant.get_qparams()
            self.weight_dtype = self.weight_qparams.dtype_meta.name
            self.act_qparams = conv_module.act_fake_quant.get_qparams()
            self.act_dtype = self.act_qparams.dtype_meta.name
        elif hasattr(conv_module.weight_observer, "get_qparams"):
            self.weight_qparams = conv_module.weight_observer.get_qparams()
            self.weight_dtype = self.weight_qparams.dtype_meta.name
            self.act_qparams = conv_module.act_observer.get_qparams()
            self.act_dtype = self.act_qparams.dtype_meta.name
        else:
            logger.error("Observer and FakeQuantize do not have get_qparams().")
        super().__init__(expr, irgraph, op_cls)

    def add_opr_out_tensors(self):
        for o in self.expr.outputs:
            out_tensor = self.resolver.get_ir_tensor(o, self.op)
            out_tensor.set_qparams_from_mge_qparams(self.act_qparams)
            out_tensor.q_dtype = self.act_dtype
            self.op.add_out_tensors(out_tensor)

    def add_const_inputs(self, weight_format):
        if self.weight is not None:
            weight_tensor = self.resolver.get_ir_tensor(
                self.weight,
                user_opr=self.op,
                name=self.expr.inputs[0]._name + "_weight",
                axis_order=weight_format,
            )
            weight_tensor.set_qparams_from_mge_qparams(self.weight_qparams)
            self.op.add_inp_tensors(weight_tensor)

        if self.bias is not None:
            bias_tensor = self.resolver.get_ir_tensor(
                self.bias, user_opr=self.op, name=self.expr.inputs[0]._name + "_bias"
            )
            bias_tensor.set_qparams(
                scale=self.op.inp_tensors[0].scale * weight_tensor.scale,
                zero_point=0,
                q_dtype="int32",
                np_dtype="int32",
            )
            self.op.add_inp_tensors(bias_tensor)


@_register_op(QAT.ConvBn2d)
class GenQConvBn2dOpr(GenQConvBnBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph, Conv2dOpr)
        self.add_opr_vars(AxisOrder.OIHW)


@_register_op(QAT.ConvBnRelu2d)
class GenQConvBnRelu2dOpr(GenQConvBnBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph, ConvRelu2dOpr)
        self.add_opr_vars(AxisOrder.OIHW)
