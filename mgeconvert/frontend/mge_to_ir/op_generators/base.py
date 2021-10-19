# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import json
from abc import ABC

from mgeconvert.converter_ir.ir_op import OpBase

from ..mge_utils import get_mge_version, get_opr_type
from ..symbolvar_resolver import SymbolVarResolver

mge_version = get_mge_version()

MGE2OP = {}


def _register_op(*ops):
    def callback(impl):
        for op in ops:
            MGE2OP[op] = impl
        return impl

    return callback


class OpGenBase(ABC):
    def __init__(self, mge_opr, irgraph) -> None:
        self.mge_opr = mge_opr
        self.irgraph = irgraph
        self.name = mge_opr.name
        self.name = self.name.replace(":", "_")
        self.name = self.name.replace(".", "_")
        self.name = self.name.replace(",", "_")

        self.id = mge_opr.id
        try:
            self.type = get_opr_type(mge_opr)
        except AssertionError:
            self.type = None
        self.flag = None
        self.params = (
            mge_opr.params if mge_version <= "0.6.0" else json.loads(mge_opr.params)
        )
        self.activation = "IDENTITY"

        self.resolver = SymbolVarResolver(irgraph)
        self.op = OpBase()

    def get_opr(self):
        return self.op

    def add_tensors(self, mge_opr):
        # set inp var
        for x in mge_opr.inputs:
            self.op.add_inp_tensors(self.resolver.get_ir_tensor(x, user_opr=self.op))
        # set out var
        for x in mge_opr.outputs:
            self.op.add_out_tensors(self.resolver.get_ir_tensor(x, owner_opr=self.op))
