# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ...converter_ir.ir_graph import IRGraph
from ...converter_ir.ir_tensor import AxisOrder, IRTensor
from .mge_utils import get_dep_vars, get_shape, get_symvar_value


class SymbolVarResolver:
    def __init__(self, irgraph: IRGraph, axis_order=AxisOrder.NCHW) -> None:
        self.irgraph = irgraph
        self.axis_order = axis_order

    def resolve(self, sym_var, owner_opr=None, axis_order=AxisOrder.NCHW):
        name = sym_var.name
        name = name.replace(":", "_")
        name = name.replace(".", "_")
        name = name.replace(",", "_")

        try:
            dtype = sym_var.dtype
        except:  # pylint: disable=bare-except
            dtype = None

        np_data = None
        try:
            if len(get_dep_vars(sym_var, "Host2DeviceCopy")) == 0:
                np_data = get_symvar_value(sym_var)
        except:  # pylint: disable=bare-except
            np_data = None

        try:
            scale = dtype.metadata["mgb_dtype"]["scale"]
        except:  # pylint: disable=bare-except
            scale = None

        try:
            zero_point = dtype.metadata["mgb_dtype"]["zero_point"]
        except:  # pylint: disable=bare-except
            zero_point = None

        return IRTensor(
            name=name,
            shape=get_shape(sym_var),
            dtype=dtype,
            scale=scale,
            zero_point=zero_point,
            np_data=np_data,
            owner_opr=owner_opr,
            axis=axis_order,
        )

    def get_ir_tensor(
        self, sym_var, owner_opr=None, user_opr=None, axis_order=AxisOrder.NCHW,
    ):
        ir_tensor = self.resolve(sym_var, owner_opr=owner_opr, axis_order=axis_order)
        ori_tensor = self.irgraph.get_tensor(sym_var.id, ir_tensor)
        if user_opr is not None and user_opr not in ori_tensor.user_opr:
            ori_tensor.add_user_opr(user_opr)
        return ori_tensor
