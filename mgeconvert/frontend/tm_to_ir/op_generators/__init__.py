# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from .base import EXPR2OP
from .batchnorm import GenBatchNormalizationOpr
from .broadcast import GenBroadcastOpr
from .concat import GenConcatOpr, GenQConcatOpr
from .constant import ConstantOpr
from .conv2d import GenConv2dOpr, GenQConv2dOpr
from .conv_bn2d import *
from .deconv import GenDeconv2dOpr, GenQDeconv2dOpr
from .dropout import GenDropoutOpr
from .elemwise import *
from .flatten import GenFlattenOpr
from .getvarshape import GenGetVarShapeOpr
from .matmul import *
from .pad import *
from .pooling import *
from .reduce import GenReduceOpr
from .reshape import GenRepeatOpr, GenReshapeOpr
from .resize import GenResizeOpr
from .softmax import GenSoftmaxOpr
from .squeeze import GenSqueezeOpr
from .subtensor import GenGetSubtensorOpr
from .transpose import GenTransposeOpr
from .typecvt import GenTypeCvtOpr
