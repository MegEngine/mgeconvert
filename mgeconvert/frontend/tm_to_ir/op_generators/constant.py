from ....converter_ir.ir_op import ConstantOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


@_register_op("Constant")
class GenConstantOpr(OpGenBase):
    name = "Constant"

    def __init__(self, expr=None, net=None) -> None:
        super().__init__(expr, net)
        self.op = ConstantOpr()
        self.add_opr_out_tensors()
