# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Sequence

import numpy as np

from .cnlib import cambriconLib as cnlib
from .tensor import Tensor


class OperatorBase:

    __name = None
    __inp_tensor: Sequence = []
    __out_tensor: Sequence = []
    _compiled = None
    _param = None
    _op = None
    _op_type = None

    def __init__(self, name, optype, inputs=None, outputs=None):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self._compiled = False
        self.param_dict = {}
        self._op_type = optype

        for x in self.inputs:
            x.ensure_mlu()

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, n):
        assert isinstance(n, str), "Invalid 'name' type{}".format(type(n))
        self.__name = n

    @property
    def inputs(self):
        return self.__inp_tensor

    @inputs.setter
    def inputs(self, inp):
        if not isinstance(inp, Sequence):
            inp = [inp]
        self.__inp_tensor = inp

    @property
    def outputs(self):
        return self.__out_tensor

    @outputs.setter
    def outputs(self, out):
        if not isinstance(out, Sequence):
            out = [out]
        self.__out_tensor = out

    @property
    def type(self):
        return self._op_type

    @property
    def compiled(self):
        return self._compiled

    @property
    def op(self):
        return self._op

    def _ensure_param(self):
        if self._param is None:
            self.make_param()

    def _ensure_made(self):
        if self._op is None:
            self.make()

    def _ensure_compiled(self):
        if not self.compiled:
            self.compile()

    def make_param(self):
        """Create param.
        Oprs should override this function."""

    def make(self):
        """Create opr.
        Oprs should not override this function."""

        self._ensure_param()
        if self._op is None:
            self.make_device()

    def make_device(self):
        """Create opr, please make it clear that host opr is not need to be created.
        Oprs should override this function."""

        raise NotImplementedError("Function was not implemented.")

    def compile(self):
        if self._op is None:
            self.make()
        cnlib.cnmlCompileBaseOp_V2(self._op)
        self._compiled = True

    def forward(self, cnqueue):
        """Run opr on device.
        Opr should not override this function."""

        self._ensure_compiled()
        self.forward_trait(cnqueue)
        cnlib.cnrtSyncQueue(cnqueue)
        for t in self.outputs:
            t.d2h()

    def forward_trait(self, cnqueue):
        """Call CNML computation API.
        Opr should override this function."""

        raise NotImplementedError("Function was not implemented.")

    def destroy_param(self):
        """Destroy param.
        Opr should override this function."""

    def destroy(self):
        """Destroy opr.
        Opr should not override this function."""

        if not self._op is None:
            cnlib.cnDestroyBaseOp(self._op)
            self._op = None
        if not self._param is None:
            self.destroy_param()
            self._param = None
        for t in self.param_dict.values():
            if isinstance(t, Tensor):
                t.destroy()
        self.param_dict.clear()


class DeviceMemcpy(OperatorBase):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, "DeviceMemcpy", inputs, outputs)

    def make_device(self):
        self._op = cnlib.cnDevMemcpyOp(
            self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeDeviceMemcpyOpForward_V4(
            self.op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class Floor(OperatorBase):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, "Floor", inputs, outputs)

    def make_device(self):
        self._op = cnlib.cnFloorOp(
            self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeFloorOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class Active(OperatorBase):
    __active_mode = None

    __active_map = {
        "NONE": cnlib.CNML_ACTIVE_NONE,
        "IDENTITY": cnlib.CNML_ACTIVE_NONE,
        "SIGMOID": cnlib.CNML_ACTIVE_SIGMOID,
        "RELU": cnlib.CNML_ACTIVE_RELU,
        "TANH": cnlib.CNML_ACTIVE_TANH,
        "RELU1": cnlib.CNML_ACTIVE_RELU1,
        "RELU6": cnlib.CNML_ACTIVE_RELU6,
        "HARD_SIGMOID": cnlib.CNML_ACTIVE_HARD_SIGMOID,
    }

    def __init__(self, name, inputs, outputs, active_mode):
        super().__init__(name, "Active", inputs, outputs)
        self.__active_mode = self.__active_map[active_mode]

    def make_device(self):
        self._op = cnlib.cnActiveOp(
            self.__active_mode, self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeActiveOpForward_V4(
            self.op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class Elemwise(OperatorBase):
    __elemwise_type = None

    __create_map = {
        "ABS": cnlib.cnAbsOp,
        "EXP": cnlib.cnExpOp,
        "LOG": cnlib.cnLogOp,
        "POW": cnlib.cnPowOp,
        "ADD": cnlib.cnAddOp,
        "SUB": cnlib.cnSubOp,
        "MUL": cnlib.cnMultOp,
        "TRUE_DIV": cnlib.cnDivOp,
        "NEGATE": cnlib.cnSubOp,
        "CYCLE_ADD": cnlib.cnCycleAddOp,
        "CYCLE_MUL": cnlib.cnCycleMultOp,
    }

    __compute_map = {
        "ABS": cnlib.cnmlComputeAbsOpForward_V4,
        "EXP": cnlib.cnmlComputeExpOpForward_V4,
        "LOG": cnlib.cnmlComputeLogOpForward_V4,
        "POW": cnlib.cnmlComputePowerOpForward_V4,
        "ADD": cnlib.cnmlComputeBroadcastAddOpForward_V4,
        "SUB": cnlib.cnmlComputeBroadcastSubOpForward_V4,
        "MUL": cnlib.cnmlComputeBroadcastMultOpForward_V4,
        "TRUE_DIV": cnlib.cnmlComputeRealDivOpForward_V4,
        "NEGATE": cnlib.cnmlComputeBroadcastSubOpForward_V4,
        "CYCLE_ADD": cnlib.cnmlComputeCycleAddOpForward_V4,
        "CYCLE_MUL": cnlib.cnmlComputeCycleMultOpForward_V4,
    }

    def __init__(self, name, inputs, outputs, elemwise_type, **kwargs):
        super().__init__(name, "Elemwise", inputs, outputs)
        self.__elemwise_type = elemwise_type
        self._kwargs = kwargs

    def make_device(self):
        if self.__elemwise_type in ["ABS", "EXP", "LOG"]:
            self._op = self.__create_map[self.__elemwise_type](
                self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
            )
        elif self.__elemwise_type == "POW":
            self._op = self.__create_map[self.__elemwise_type](
                self.inputs[0].cnmlTensor,
                self.outputs[0].cnmlTensor,
                self._kwargs["power"],
            )
        else:
            self._op = self.__create_map[self.__elemwise_type](
                self.inputs[0].cnmlTensor,
                self.inputs[1].cnmlTensor,
                self.outputs[0].cnmlTensor,
            )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        if self.__elemwise_type in ["ABS", "EXP", "LOG", "POW"]:
            self.__compute_map[self.__elemwise_type](
                self.op,
                None,
                self.inputs[0].mludata,
                None,
                self.outputs[0].mludata,
                cnqueue,
                None,
            )
        else:
            self.__compute_map[self.__elemwise_type](
                self.op,
                None,
                self.inputs[0].mludata,
                None,
                self.inputs[1].mludata,
                None,
                self.outputs[0].mludata,
                cnqueue,
                None,
            )


class Reduce(OperatorBase):
    __dim = None
    __reduce_type = None

    __create_map = {
        "AND": cnlib.cnReduceAndOp,
        "OR": cnlib.cnReduceOrOp,
        "SUM": cnlib.cnReduceSumOp,
        "MAX": cnlib.cnReduceMaxOp,
        "MEAN": cnlib.cnReduceMeanOp,
        "PRODUCT": cnlib.cnReduceProductOp,
    }

    __compute_map = {
        "AND": cnlib.cnmlComputeReduceAndOpForward,
        "OR": cnlib.cnmlComputeReduceOrOpForward,
        "SUM": cnlib.cnmlComputeReduceSumOpForward_V4,
        "MAX": cnlib.cnmlComputeReduceMaxOpForward_V4,
        "MEAN": cnlib.cnmlComputeReduceMeanOpForward_V4,
        "PRODUCT": cnlib.cnmlComputeReduceProductOpForward_V2,
    }

    def __init__(self, name, inputs, outputs, dim, reduce_type):
        super().__init__(name, "Reduce", inputs, outputs)
        self.__dim = dim
        self.__reduce_type = reduce_type

    def make_device(self):
        self._op = self.__create_map[self.__reduce_type](
            self.__dim, self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        self.__compute_map[self.__reduce_type](
            self.op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


# original bn in cambricon: y = (x - running_mean) * running_var
# shifted to: y = (x - running_mean) / sqrt(running_var)
class BatchNorm(OperatorBase):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, "BatchNorm", inputs, outputs)

    def _deal_param(self):
        running_var = self.param_dict["running_var"].cpudata
        with np.errstate(divide="ignore"):
            self.param_dict["running_var"].cpudata = 1.0 / np.sqrt(running_var)

    def make_device(self):
        assert "running_mean" in self.param_dict.keys(), "need mean."
        assert "running_var" in self.param_dict.keys(), "need var."
        self._deal_param()
        self.param_dict["running_mean"].ensure_mlu()
        self.param_dict["running_var"].ensure_mlu()
        self._op = cnlib.cnBatchNormOp(
            self.inputs[0].cnmlTensor,
            self.outputs[0].cnmlTensor,
            self.param_dict["running_mean"].cnmlTensor,
            self.param_dict["running_var"].cnmlTensor,
        )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeBatchNormOpForward_V4(
            self.op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class MatMul(OperatorBase):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, "MatMul", inputs, outputs)
        self._quantize_param = []

    def make_device(self):

        self.param_dict["weight"].ensure_mlu()

        if "bias" in self.param_dict.keys():
            self.param_dict["bias"].ensure_mlu()
            self._op = cnlib.cnMatMulOp(
                self.inputs[0].cnmlTensor,
                self.outputs[0].cnmlTensor,
                self.param_dict["weight"].cnmlTensor,
                self.param_dict["bias"].cnmlTensor,
            )
        else:
            self._op = cnlib.cnMatMulOp(
                self.inputs[0].cnmlTensor,
                self.outputs[0].cnmlTensor,
                self.param_dict["weight"].cnmlTensor,
                None,
            )

        input_quant = cnlib.cnQuantizedParam(
            self.inputs[0]._cn_position, self.inputs[0]._cn_scale, 0
        )
        self._quantize_param.append(input_quant)
        cnlib.cnmlSetOperationComputingDataType(
            self._op, self.inputs[0].cnmlTensor, cnlib.CNML_DATA_INT8, input_quant
        )
        if self.param_dict["weight"]._cn_data_type == cnlib.CNML_DATA_FLOAT32:
            filter_quant = cnlib.cnQuantizedParam(
                self.param_dict["weight"]._cn_position,
                self.param_dict["weight"]._cn_scale,
                0,
            )
            self._quantize_param.append(filter_quant)
            cnlib.cnmlSetOperationComputingDataType(
                self._op,
                self.param_dict["weight"].cnmlTensor,
                cnlib.CNML_DATA_INT8,
                filter_quant,
            )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeMlpOpForward_V4(
            self.op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )

    def destroy_param(self):
        for quantize_param in self._quantize_param:
            cnlib.cnDestroyQuantizedParam(quantize_param)
        self._quantize_param.clear()


class Conv(OperatorBase):
    __stride = None
    __dilation = None
    __pad = None

    def __init__(
        self,
        name,
        inputs,
        outputs,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        pad_h,
        pad_w,
        groups=1,
    ):
        super().__init__(name, "Conv", inputs, outputs)
        self.__stride = [stride_h, stride_w]
        self.__dilation = [dilation_h, dilation_w]
        self.__pad = [pad_h, pad_w]
        self.groups = groups
        self._quantize_param = []

    @property
    def stride(self):
        return self.__stride

    @property
    def pad(self):
        return self.__pad

    def make_param(self):
        self._param = cnlib.cnConvOpParam(
            self.__stride[0],
            self.__stride[1],
            self.__dilation[0],
            self.__dilation[1],
            self.__pad[0] * 2,
            self.__pad[1] * 2,
        )

    def make_device(self):
        assert "W" in self.param_dict.keys(), "Need Filter."
        assert "B" in self.param_dict.keys(), "Need Bias."
        self.param_dict["W"].ensure_mlu()
        self.param_dict["B"].ensure_mlu()

        self._ensure_param()
        self._op = cnlib.cnConvOp(
            self._param,
            self.inputs[0].cnmlTensor,
            self.outputs[0].cnmlTensor,
            self.param_dict["W"].cnmlTensor,
            self.param_dict["B"].cnmlTensor,
            self.groups,
        )

        input_quant = cnlib.cnQuantizedParam(
            self.inputs[0]._cn_position, self.inputs[0]._cn_scale, 0
        )
        self._quantize_param.append(input_quant)
        cnlib.cnmlSetOperationComputingDataType(
            self._op, self.inputs[0].cnmlTensor, cnlib.CNML_DATA_INT8, input_quant
        )

        if self.param_dict["W"]._cn_data_type == cnlib.CNML_DATA_FLOAT32:
            filter_quant = cnlib.cnQuantizedParam(
                self.param_dict["W"]._cn_position, self.param_dict["W"]._cn_scale, 0
            )
            self._quantize_param.append(filter_quant)
            cnlib.cnmlSetOperationComputingDataType(
                self._op,
                self.param_dict["W"].cnmlTensor,
                cnlib.CNML_DATA_INT8,
                filter_quant,
            )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        if self.groups == 1:
            cnlib.cnmlComputeConvOpForward_V4(
                self._op,
                None,
                self.inputs[0].mludata,
                None,
                self.outputs[0].mludata,
                cnqueue,
                None,
            )
        else:
            cnlib.cnmlComputeConvOpForward_V4(
                self._op,
                None,
                self.inputs[0].mludata,
                None,
                self.outputs[0].mludata,
                cnqueue,
                None,
            )

    def destroy_param(self):
        cnlib.cnDestroyConvOpParam(self._param)
        self._param = None
        for quantize_param in self._quantize_param:
            cnlib.cnDestroyQuantizedParam(quantize_param)
        self._quantize_param.clear()


class Reshape(OperatorBase):
    __shape = None

    def __init__(self, name, inputs, outputs):
        super().__init__(name, "Reshape", inputs, outputs)
        self.__shape = self.outputs[0].shape

    def make_param(self):
        self._param = cnlib.cnReshapeOpParam(
            self.__shape[0],
            self.__shape[1],
            self.__shape[2],
            self.__shape[3],
            self.inputs[0]._cn_tensor_layout,
        )

    def make_device(self):
        self._op = cnlib.cnReshapeOp(
            self._param, self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeReshapeOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class Pool(OperatorBase):
    _map_pool_mode = {
        "MAX": cnlib.CNML_POOL_MAX,
        "AVG": cnlib.CNML_POOL_AVG,
    }

    __windows = None
    __stride = None
    __pad = None
    __dilation = None
    __pool_mode = None
    __strategy_mode = None
    __real = None

    def __init__(
        self,
        name,
        inputs,
        outputs,
        windows_h,
        windows_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        pool_mode="MAX",
        strategy_mode=cnlib.CNML_POOL_KVALID,
    ):
        super().__init__(name, "Pool", inputs, outputs)
        self.__windows = [windows_h, windows_w]
        self.__stride = [stride_h, stride_w]
        self.__pad = [pad_h, pad_w]
        self.__dilation = [dilation_h, dilation_w]
        self.__pool_mode = self._map_pool_mode[pool_mode]
        self.__strategy_mode = strategy_mode
        self.__real = pool_mode == "MAX"

    def make_param(self):
        self._param = cnlib.cnPoolOpParam(
            self.__windows[0],
            self.__windows[1],
            self.__stride[0],
            self.__stride[1],
            self.__pad[0] * 2,
            self.__pad[1] * 2,
            self.__dilation[0],
            self.__dilation[1],
            self.__pool_mode,
            self.__strategy_mode,
            self.__real,
        )

    def make_device(self):
        self._ensure_param()
        self._op = cnlib.cnPoolOp(
            self._param, self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputePoolOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )

    def destroy_param(self):
        cnlib.cnDestroyPoolOpParam(self._param)


class Concat(OperatorBase):
    _map_concat_dim = {
        0: cnlib.CNML_DIM_N,
        1: cnlib.CNML_DIM_C,
        2: cnlib.CNML_DIM_H,
        3: cnlib.CNML_DIM_W,
    }

    __mode = None

    def __init__(self, name, inputs, outputs, mode=1):
        super().__init__(name, "Concat", inputs, outputs)
        self.__mode = self._map_concat_dim[mode]

    def make_param(self):
        self._param = cnlib.cnConcatOpParam(
            len(self.inputs), len(self.outputs), self.__mode
        )

    def make_device(self):
        self._ensure_param()
        inputs = cnlib.VectorcnmlTensor([x.cnmlTensor for x in self.inputs])
        outputs = cnlib.VectorcnmlTensor([x.cnmlTensor for x in self.outputs])
        self._op = cnlib.cnConcatOp(self._param, inputs, outputs)
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        inputs = cnlib.Vectorvoid([x.mludata for x in self.inputs])
        outputs = cnlib.Vectorvoid([x.mludata for x in self.outputs])
        cnlib.cnComputeConcatOp(self._op, None, inputs, None, outputs, cnqueue, None)


class Slice(OperatorBase):
    def __init__(self, name, inputs, outputs, slc):
        super().__init__(name, "Slice", inputs, outputs)
        self.nb, self.ne, self.ns = list(map(int, slc[0]))
        self.cb, self.ce, self.cs = list(map(int, slc[1]))
        self.hb, self.he, self.hs = list(map(int, slc[2]))
        self.wb, self.we, self.ws = list(map(int, slc[3]))

    def make_param(self):
        # create op param
        self._param = cnlib.cnSliceOpParam(
            self.nb,
            self.cb,
            self.hb,
            self.wb,
            self.ne,
            self.ce,
            self.he,
            self.we,
            self.ns,
            self.cs,
            self.hs,
            self.ws,
        )

    def make_device(self):
        # create op
        self._op = cnlib.cnSliceOp(
            self._param, self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )
        # set computing data type.
        # set layout
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeStridedSliceOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class Broadcast(OperatorBase):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, "Broadcast", inputs, outputs)

    def make_param(self):
        # create op param
        pass

    def make_device(self):
        # create op
        self._op = cnlib.cnBroadcastOp(
            self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeBroadcastOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class BatchMatMul(OperatorBase):
    __trans_a = None
    __trans_b = None

    def __init__(self, name, inputs, outputs, trans_a=False, trans_b=False):
        super().__init__(name, "BatchMatMul", inputs, outputs)
        self._quantize_param = []
        self.__trans_a, self.__trans_b = trans_a, trans_b

    def make_param(self):
        # create op param
        pass

    def make_device(self):
        # create op
        self._op = cnlib.cnBatchDotOp(
            self.inputs[0].cnmlTensor,
            self.inputs[1].cnmlTensor,
            self.outputs[0].cnmlTensor,
            self.__trans_a,
            self.__trans_b,
        )
        cnlib.cnmlSetOperationComputingDataType(
            self._op, self.inputs[0].cnmlTensor, cnlib.CNML_DATA_INT8, None
        )
        cnlib.cnmlSetOperationComputingDataType(
            self._op, self.inputs[1].cnmlTensor, cnlib.CNML_DATA_INT8, None
        )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeBatchDotOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.inputs[1].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class Dimshuffle(OperatorBase):
    __pattern = None

    def __init__(self, name, inputs, outputs, pattern):
        super().__init__(name, "Dimshuffle", inputs, outputs)
        self.__pattern = pattern

    def make_param(self):
        # create op param
        self._param = cnlib.cnTransposeOpParam(
            self.__pattern[0],
            self.__pattern[1],
            self.__pattern[2],
            self.__pattern[3],
            self.inputs[0]._cn_tensor_layout,
        )

    def make_device(self):
        # create op
        self._op = cnlib.cnTransposeProOp(
            self._param, self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeTransposeProOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class Sqrt(OperatorBase):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, "Sqrt", inputs, outputs)

    def make_device(self):
        self._op = cnlib.cnSqrtOp(self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor)

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeSqrtOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class Rsqrt(OperatorBase):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, "Rsqrt", inputs, outputs)

    def make_device(self):
        self._op = cnlib.cnRsqrtOp(
            self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeRsqrtOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class Square(OperatorBase):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, "Square", inputs, outputs)

    def make_device(self):
        self._op = cnlib.cnSquareOp(
            self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeSquareOpForward_V2(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class BasicDiv(OperatorBase):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, "BasicDiv", inputs, outputs)

    def make_device(self):
        self._op = cnlib.cnBasicDivOp(
            self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeBasicDivOpForward(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class AddPad(OperatorBase):
    def __init__(self, name, inputs, outputs, pt, pb, pl, pr, *_):
        super().__init__(name, "AddPad", inputs, outputs)
        self._pt = pt
        self._pb = pb
        self._pl = pl
        self._pr = pr

    def make_param(self):
        self._param = cnlib.cnAddPadOpParam(self._pt, self._pb, self._pl, self._pr)

    def make_device(self):
        self._op = cnlib.cnAddPadOp(
            self._param, self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor
        )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeAddPadOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class Deconv(OperatorBase):
    __stride = None
    __dilation = None
    __pad = None

    def __init__(
        self,
        name,
        inputs,
        outputs,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        pad_h,
        pad_w,
    ):
        super().__init__(name, "Deconv", inputs, outputs)
        self.__stride = [stride_h, stride_w]
        self.__dilation = [dilation_h, dilation_w]
        self.__pad = [pad_h, pad_w]

    @property
    def stride(self):
        return self.__stride

    @property
    def pad(self):
        return self.__pad

    @property
    def dilation(self):
        return self.__dilation

    def make_param(self):
        self._param = cnlib.cnDeconvOpParam(
            self.__stride[0],
            self.__stride[1],
            self.__dilation[0],
            self.__dilation[1],
            self.__pad[0] * 2,
            self.__pad[1] * 2,
        )

    def make_device(self):
        assert "W" in self.param_dict.keys(), "Need Filter"
        assert "B" in self.param_dict.keys(), "Need Bias"
        self.param_dict["W"].ensure_mlu()
        self.param_dict["B"].ensure_mlu()

        self._ensure_param()
        self._op = cnlib.cnDeconvOp(
            self._param,
            self.inputs[0].cnmlTensor,
            self.outputs[0].cnmlTensor,
            self.param_dict["W"].cnmlTensor,
            self.param_dict["B"].cnmlTensor,
        )

        input_quant = cnlib.cnQuantizedParam(
            self.inputs[0]._cn_position, self.inputs[0]._cn_scale, 0
        )
        cnlib.cnmlSetOperationComputingDataType(
            self._op, self.inputs[0].cnmlTensor, cnlib.CNML_DATA_INT8, input_quant
        )
        if self.param_dict["W"]._cn_data_type == cnlib.CNML_DATA_FLOAT32:
            filter_quant = cnlib.cnQuantizedParam(
                self.param_dict["W"]._cn_position, self.param_dict["W"]._cn_scale, 0
            )
            cnlib.cnmlSetOperationComputingDataType(
                self._op,
                self.param_dict["W"].cnmlTensor,
                cnlib.CNML_DATA_INT8,
                filter_quant,
            )
        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeDeconvOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )

    def destroy_param(self):
        cnlib.cnDestroyDeconvOpParam(self._param)
        self._param = None


class ConvFirst(OperatorBase):
    __stride = None
    __dilation = None
    __pad = None

    def __init__(
        self,
        name,
        inputs,
        outputs,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        pad_h,
        pad_w,
    ):
        super().__init__(name, "ConvFirst", inputs, outputs)
        self.__stride = [stride_h, stride_w]
        self.__dilation = [dilation_h, dilation_w]
        self.__pad = [pad_h, pad_w]

    @property
    def stride(self):
        return self.__stride

    @property
    def dilation(self):
        return self.__dilation

    @property
    def pad(self):
        return self.__pad

    def make_param(self):
        self._param = cnlib.cnConvFirstOpParam(
            self.__stride[0],
            self.__stride[1],
            self.__dilation[0],
            self.__dilation[1],
            self.__pad[0],
            self.__pad[1],
        )

    def make_device(self):
        assert "W" in self.param_dict.keys(), "Need Filter"
        assert "B" in self.param_dict.keys(), "Need Bias"
        assert "mean" in self.param_dict.keys(), "Need mean"
        assert "std" in self.param_dict.keys(), "Need std"
        self.param_dict["W"].ensure_mlu()
        self.param_dict["B"].ensure_mlu()
        self.param_dict["mean"].ensure_mlu()
        self.param_dict["std"].ensure_mlu()

        self._ensure_param()
        self._op = cnlib.cnConvFirstOp(
            self._param,
            self.inputs[0].cnmlTensor,
            self.outputs[0].cnmlTensor,
            self.param_dict["W"].cnmlTensor,
            self.param_dict["B"].cnmlTensor,
            self.param_dict["mean"].cnmlTensor,
            self.param_dict["std"].cnmlTensor,
        )

        cnlib.cnmlSetOperationComputingLayout(
            self._op, self.inputs[0]._cn_tensor_layout
        )

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeConvFirstOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )

    def destroy_param(self):
        cnlib.cnDestroyConvFirstOpParam(self._param)
        self._param = None


class Cast(OperatorBase):
    _map_cast_type = {
        "fp16-fp32": cnlib.CNML_CAST_FLOAT16_TO_FLOAT32,
        "fp32-fp16": cnlib.CNML_CAST_FLOAT32_TO_FLOAT16,
    }

    def __init__(self, name, inputs, outputs, cast_type):
        super().__init__(name, "Cast", inputs, outputs)
        self.__cast_type = Cast._map_cast_type[cast_type]

    def make_device(self):
        self._op = cnlib.cnCastOp(self.inputs[0].cnmlTensor, self.outputs[0].cnmlTensor)

    def forward_trait(self, cnqueue):
        cnlib.cnmlComputeCastOpForward_V4(
            self._op,
            None,
            self.inputs[0].mludata,
            None,
            self.outputs[0].mludata,
            cnqueue,
            None,
        )


class Fusion(OperatorBase):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, "Fusion", inputs, outputs)

    def make_device(self):
        self._op = cnlib.cnFusionOp()

    def set_core_num(self, num):
        cnlib.cnmlSetFusionOpCorenum(self._op, num)

    def fuse_op(self, base_op):
        self._ensure_made()
        assert not base_op.compiled, "The opr should not be compiled before fused!"
        assert (
            not self._compiled
        ), "Attempting fuse an opr after the fusion opr was compiled is ambitious."
        base_op.make()
        cnlib.cnmlFuseOp(base_op.op, self._op)

    def set_fusion_io(self):
        self._ensure_made()
        assert not self._compiled
        inputs = cnlib.VectorcnmlTensor([x.cnmlTensor for x in self.inputs])
        outputs = cnlib.VectorcnmlTensor([x.cnmlTensor for x in self.outputs])
        cnlib.cnFusionIO(self._op, inputs, outputs)

    def add_fusion_input(self, inp):
        self._ensure_made()
        assert not self._compiled
        cnlib.cnmlAddFusionInput(self._op, inp.cnmlTensor)

    def add_fusion_output(self, oup):
        self._ensure_made()
        assert not self._compiled
        cnlib.cnmlAddFusionOutput(self._op, oup.cnmlTensor)

    def compile(self):
        self._ensure_made()
        cnlib.cnmlCompileFusionOp_V2(self._op)
        self._compiled = True

    def forward_trait(self, cnqueue):
        input_tensors = cnlib.VectorcnmlTensor([x.cnmlTensor for x in self.inputs])
        output_tensors = cnlib.VectorcnmlTensor([x.cnmlTensor for x in self.outputs])

        inputs = cnlib.Vectorvoid([x.mludata for x in self.inputs])
        outputs = cnlib.Vectorvoid([x.mludata for x in self.outputs])

        cnlib.cnComputeFusionOp(
            self._op, input_tensors, inputs, output_tensors, outputs, cnqueue, None
        )

    def destroy(self):
        if self._op is not None:
            cnlib.cnDestroyFusionOp(self._op)
            self._op = None
