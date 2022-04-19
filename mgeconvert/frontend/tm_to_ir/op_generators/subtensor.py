# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module

from typing import List  # pylint: disable=unused-import
from typing import Sequence

import numpy as np
from megengine.traced_module.expr import CallMethod

from ....converter_ir.ir_op import GetSubTensorOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


@_register_op("__getitem__")
class GenGetSubtensorOpr(OpGenBase):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph)
        assert isinstance(self.expr, CallMethod)
        self.axis = []  # type: List[int]
        self.begin_params = []  # type: List[int]
        self.end_params = []  # type: List[int]
        self.step_params = []  # type: List[int]
        self.squeeze_axis = []  # type: List[int]
        if isinstance(self.args[1], Sequence):
            for i in range(len(self.args[1])):
                slice = self.args[1][i]
                if isinstance(slice, int):
                    start = slice
                    stop = slice + 1
                    step = 1
                    self.squeeze_axis.append(i)
                elif slice.start or slice.stop:
                    start = slice.start if slice.start is not None else 0
                    stop = (
                        slice.stop if slice.stop is not None else np.iinfo(np.int32).max
                    )
                    step = slice.step if slice.step is not None else 1
                else:
                    continue
                self.begin_params.append(start)
                self.end_params.append(stop)
                self.step_params.append(step)
                self.axis.append(i)
        elif isinstance(self.args[1], int):
            start = self.args[1]
            stop = self.args[1] + 1
            step = 1
            self.squeeze_axis.append(0)
            self.begin_params.append(start)
            self.end_params.append(stop)
            self.step_params.append(step)
            self.axis.append(0)

        self.op = GetSubTensorOpr(
            self.axis,
            self.begin_params,
            self.end_params,
            self.step_params,
            self.squeeze_axis,
        )
        self.add_opr_vars()

    def add_opr_vars(self):
        inp = self.args[0]
        inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
        self.op.add_inp_tensors(inp_tensor)
        self.add_opr_out_tensors()
        if (
            hasattr(self.op.inp_tensors[0], "scale")
            and self.op.inp_tensors[0].scale is not None
        ):
            for o in self.op.out_tensors:
                o.set_qparams_from_other_tensor(self.op.inp_tensors[0])
