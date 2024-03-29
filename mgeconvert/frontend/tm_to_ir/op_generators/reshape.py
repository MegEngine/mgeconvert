# pylint: disable=import-error,no-name-in-module,no-member

import megengine.functional as F
from megengine.traced_module.expr import CallFunction, CallMethod

from ....converter_ir.ir_op import RepeatOpr, ReshapeOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


@_register_op(F.reshape, "reshape")
class GenReshapeOpr(OpGenBase):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph)
        if isinstance(self.expr, CallFunction):
            self.out_shape = self.args[1]
            self.op = ReshapeOpr(self.out_shape)
        elif isinstance(self.expr, CallMethod):
            self.out_shape = self.expr.outputs[0].shape
            self.op = ReshapeOpr(self.out_shape)
        self.add_opr_vars()

    def add_opr_vars(self):
        inp = self.args[0]
        inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
        self.op.add_inp_tensors(inp_tensor)
        if isinstance(self.expr, CallMethod):
            if len(self.args) > 1:
                for inp in self.args[1:]:
                    inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
                    self.op.add_inp_tensors(inp_tensor)
        self.add_opr_out_tensors()


@_register_op(F.repeat)
class GenRepeatOpr(OpGenBase):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph)
        assert isinstance(self.expr, CallFunction)
        self.repeats = self.args[1]
        self.axis = self.args[2]
        self.op = RepeatOpr(self.repeats, self.axis)
        self.add_opr_vars()

    def add_opr_vars(self):
        inp = self.args[0]
        inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
        self.op.add_inp_tensors(inp_tensor)
        self.add_opr_out_tensors()
