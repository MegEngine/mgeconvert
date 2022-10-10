import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.traced_module as tm
import mgeconvert.converter_ir.ir_op as Op
import numpy as np
from mgeconvert.converter_ir.ir_transform import IRTransform, TransformerRule
from mgeconvert.frontend.tm_to_ir import TM_FrontEnd


def get_ir_graph(tmodel):
    tm_resolver = TM_FrontEnd(tmodel, outspec=None)
    irgraph = tm_resolver.resolve()
    return irgraph


def check_ir_graph_correctness(irgraph):
    generated_tensors = set(irgraph.graph_inputs)
    graph_outputs = set(irgraph.graph_outputs)
    graph_inputs = set(irgraph.graph_inputs)
    for op in irgraph.all_oprs:
        for inp in op.inp_tensors:
            assert (
                inp in generated_tensors or inp.np_data is not None
            ), "inp {} used by op {} doesn't belong to this irgraph".format(inp, op)
            if inp in graph_inputs:
                graph_inputs.remove(inp)
        for oup in op.out_tensors:
            assert oup not in generated_tensors
            generated_tensors.add(oup)
            if oup in graph_outputs:
                graph_outputs.remove(oup)

    assert not graph_inputs, "Has unused inputs in irgraph"
    assert not graph_outputs, "Outputs {} don't belong to this irgraph".format(
        graph_outputs
    )


def test_remove_identity():
    class test_model(M.Module):
        def __init__(self):
            super().__init__()
            self.identity = M.Identity()

        def forward(self, x):
            y = x + x
            return self.identity(y)

    model = test_model()
    tmodel = tm.trace_module(model, mge.tensor(1))
    irgraph = get_ir_graph(tmodel)
    transformer = IRTransform([TransformerRule.REMOVE_IDENTITY])
    rst = transformer.transform(irgraph)
    check_ir_graph_correctness(rst)
    assert len(rst.all_oprs) == 1 and isinstance(rst.all_oprs[0], Op.AddOpr)


def test_fuse_activation():
    class test_model(M.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = M.Conv2d(3, 3, 3)
            self.conv2 = M.Conv2d(3, 3, 3)

        def forward(self, x):
            y = self.conv1(x)
            y = F.relu(y)
            y = self.conv2(y)
            a = y + 1
            b = F.relu(y)
            return a + b

    model = test_model()
    tmodel = tm.trace_module(model, mge.tensor(np.random.random((1, 3, 32, 32))))
    irgraph = get_ir_graph(tmodel)
    transformer = IRTransform([TransformerRule.FUSE_ACTIVATION])
    rst = transformer.transform(irgraph)
    check_ir_graph_correctness(rst)
    assert rst.all_oprs[0].activation == "RELU"
    assert rst.all_oprs[1].activation == "IDENTITY"
