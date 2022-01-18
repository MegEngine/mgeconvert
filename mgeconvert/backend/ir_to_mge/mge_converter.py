# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Dict, List, Union

import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.traced_module as tm
import numpy as np
from megengine import jit

from ...converter_ir.ir_quantizer import IRQuantizer
from .mge_op import MGE_MODULE_SUPPORT, ONNX2MGE, PARAMEXTRACT


def for_each_opr(
    all_oprs,
    inputs,
    dtypes,
    shapes,
    datas,
    outputs,
    params,
    quantizer,
    map_ir_tensor_2_mge_tensor,
    map_op_name_2_mge_module,
):
    for (
        opr,
        opr_inputs,
        input_dtypes,
        input_shapes,
        input_data,
        opr_outputs,
        param,
    ) in zip(all_oprs, inputs, dtypes, shapes, datas, outputs, params):
        converter_cls = ONNX2MGE.get(opr, None)
        assert converter_cls is not None, f"Mge Opr : {opr.name} is not supported"
        opr_convert = converter_cls(opr, param, quantizer)
        opr_convert.convert(
            opr_inputs,
            input_dtypes,
            input_shapes,
            input_data,
            opr_outputs,
            map_ir_tensor_2_mge_tensor,
            map_op_name_2_mge_module,
        )


class ONNXModule(M.Module):
    def __init__(
        self,
        all_oprs,
        inputs,
        dtypes,
        shapes,
        datas,
        outputs,
        params,
        quantizer,
        map_ir_tensor_2_mge_tensor,
        arg_names,
        map_op_name_2_mge_module,
        graph_outputs,
    ):
        super().__init__()
        self.all_oprs = all_oprs
        self.inputs = inputs
        self.dtypes = dtypes
        self.shapes = shapes
        self.datas = datas
        self.outputs = outputs
        self.param = params
        self.require_quantize = False
        self.param_fake_quant = False
        self.quant_params = {}
        self.has_quantizer = quantizer != None
        if self.has_quantizer:
            self.require_quantize = quantizer.require_quantize
            self.param_fake_quant = quantizer.param_fake_quant
            self.quant_params = quantizer.quant_params
        self.map_ir_tensor_2_mge_tensor = map_ir_tensor_2_mge_tensor
        self.arg_names = arg_names
        self.map_op_name_2_mge_module = map_op_name_2_mge_module
        self.graph_outputs = graph_outputs

    def forward(self, *args):
        assert len(args) == len(self.arg_names), "Number of inputs mismatch"
        for name, data in zip(self.arg_names, args):
            self.map_ir_tensor_2_mge_tensor[name] = data

        if self.has_quantizer:
            quantizer = IRQuantizer(self.require_quantize, self.param_fake_quant)
            quantizer.quant_params = self.quant_params
        else:
            quantizer = None

        for_each_opr(
            self.all_oprs,
            self.inputs,
            self.dtypes,
            self.shapes,
            self.datas,
            self.outputs,
            self.param,
            quantizer,
            self.map_ir_tensor_2_mge_tensor,
            self.map_op_name_2_mge_module,
        )
        return [self.map_ir_tensor_2_mge_tensor[name] for name in self.graph_outputs]


class MGEConverter:
    def __init__(self, ir_graph, quantizer: Union[IRQuantizer, None] = None):
        self.ir_graph = ir_graph
        self.quantizer = quantizer
        self.map_ir_tensor_2_mge_tensor = {}  # type: Dict[str, mge.tensor]
        self.map_op_name_2_mge_module = {}  # type: Dict[str, mge.module]
        self.tm = None

        self.arg_names = []  # type: List[str]
        self.inp_data = []  # type: List[mge.tensor]
        for input in ir_graph.graph_inputs:
            assert input.shape is not None and input.dtype is not None
            mge_input = F.zeros(shape=input.shape, dtype=input.dtype)
            self.inp_data.append(mge_input)
            self.arg_names.append(input.name)

        self.graph_outputs = [o.name for o in ir_graph.graph_outputs]

    def _construct_tm(self):
        if self.tm is not None:
            return self.tm
        all_opr = []
        inputs = []
        dtypes = []
        shapes = []
        datas = []
        outputs = []
        params = []
        for opr in self.ir_graph.all_oprs:
            op_cls = PARAMEXTRACT.get(type(opr), None)
            if op_cls is not None:
                params.append(op_cls(opr).extract())
            else:
                params.append({})
            all_opr.append(type(opr))
            inputs.append([i.name for i in opr.inp_tensors])
            dtypes.append([i.dtype for i in opr.inp_tensors])
            shapes.append([i.shape for i in opr.inp_tensors])
            datas.append([i.np_data for i in opr.inp_tensors])
            assert (
                len(opr.out_tensors) == 1
            ), "MegEngine Cannot supports multiple outputs of one Opr"
            outputs.append([o.name for o in opr.out_tensors])

            # gen module
            op_cls = MGE_MODULE_SUPPORT.get(type(opr), None)
            if op_cls is not None:
                op_cls(opr).mge_module_gen(
                    [i.name for i in opr.inp_tensors],
                    [o.name for o in opr.out_tensors],
                    self.map_op_name_2_mge_module,
                )

        module = ONNXModule(
            all_opr,
            inputs,
            dtypes,
            shapes,
            datas,
            outputs,
            params,
            self.quantizer,
            self.map_ir_tensor_2_mge_tensor,
            self.arg_names,
            self.map_op_name_2_mge_module,
            self.graph_outputs,
        )

        self.tm = tm.trace_module(module, *self.inp_data)
        return self.tm

    def dump_mge_model(self, mge_path):
        traced_module = self._construct_tm()

        @jit.trace(symbolic=True, capture_as_const=True)
        def infer_func(*args):
            return traced_module(*args)

        _ = infer_func(*self.inp_data)
        infer_func.dump(
            mge_path, arg_names=self.arg_names, optimize_for_inference=False
        )

    def dump_tm_model(self, tm_path):
        traced_module = self._construct_tm()
        mge.save(traced_module, tm_path)

    def eval(self, seed: int):
        np.random.seed(seed)
        traced_module = self._construct_tm()
        inputs = []
        for input in self.ir_graph.graph_inputs:
            assert input.shape is not None and input.dtype is not None
            np_data = np.random.randint(
                low=0, high=100, size=input.shape, dtype="int32"
            ).astype(input.dtype)
            inputs.append(mge.tensor(np_data))
        traced_module.eval()
        return traced_module(*inputs)
