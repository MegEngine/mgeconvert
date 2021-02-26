# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .mge_utils import get_dep_vars, get_shape, get_symvar_value


class FakeSymbolVar:
    def __init__(self, sid, name, shape, dtype, owner, byte_list=None):
        self.id = sid
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.owner = owner
        self.byte_list = byte_list


class Tensor:
    def __init__(self, sym_var, owner_opr):
        self.is_faked = type(sym_var) == FakeSymbolVar
        self.id = sym_var.id
        self._var = sym_var
        self.name = sym_var.name
        self.name = self.name.replace(":", "_")
        self.name = self.name.replace(".", "_")
        self.name = self.name.replace(",", "_")
        self.shape = get_shape(sym_var)
        self.owner_opr = owner_opr
        if self.shape is not None:
            self.ndim = len(self.shape)
        try:
            self.dtype = sym_var.dtype
        except:  # pylint: disable=bare-except
            self.dtype = None

        self.byte_list = getattr(sym_var, "byte_list", None)

        self.np_data = None
        try:
            if len(get_dep_vars(sym_var, "Host2DeviceCopy")) == 0:
                self.np_data = get_symvar_value(sym_var)
        except:  # pylint: disable=bare-except
            self.np_data = None

        try:
            self.qbit = self.dtype.metadata["mgb_dtype"]["name"]
        except:  # pylint: disable=bare-except
            self.qbit = None

        try:
            self.scale = self.dtype.metadata["mgb_dtype"]["scale"]
        except:  # pylint: disable=bare-except
            self.scale = None

    def set_owner_opr(self, owner_opr):
        self.owner_opr = owner_opr
