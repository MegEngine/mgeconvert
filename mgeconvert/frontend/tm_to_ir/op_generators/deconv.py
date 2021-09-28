# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=no-member

import megengine.functional as F
import megengine.module as M
import megengine.module.qat as QAT

from ....converter_ir.ir_op import Deconv2dOpr
from ....converter_ir.ir_tensor import AxisOrder
from ..tm_utils import get_logger
from .base import _register_op
from .conv2d import GenConvBase, GenQConvBase

logger = get_logger(__name__)


@_register_op(M.ConvTranspose2d, F.conv_transpose2d)
class GenDeconv2dOpr(GenConvBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph, Deconv2dOpr)
        self.add_opr_vars(AxisOrder.IOHW)


@_register_op(QAT.ConvTranspose2d)
class GenQDeconv2dOpr(GenQConvBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph, Deconv2dOpr)
        self.add_opr_vars(AxisOrder.IOHW)
