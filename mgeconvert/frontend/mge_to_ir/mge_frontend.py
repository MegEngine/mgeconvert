# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ...converter_ir.ir_graph import IRGraph
from .mge_utils import (
    get_dep_vars,
    get_opr_type,
    get_oprs_seq,
    load_comp_graph_from_file,
)
from .op_generators import MGE2OP
from .symbolvar_resolver import SymbolVarResolver


class MGE_FrontEnd:
    def __init__(self, model_path, outspec=None, prune_reshape=True):
        _, outputs = load_comp_graph_from_file(model_path)
        if outspec is not None:
            output_spec = outspec.copy()
            all_vars = get_dep_vars(outputs) + outputs
            new_outputs = {}
            for i in all_vars:
                if i.name in output_spec:
                    new_outputs[i.name] = i
                    output_spec.remove(i.name)
            assert len(output_spec) == 0, "Can not find {} in this model".format(
                output_spec
            )
            outputs = [new_outputs[i] for i in outspec]
        self.ori_all_oprs = get_oprs_seq(outputs, prune_reshape=prune_reshape)
        self._orig_inputs = []
        self._orig_outputs = outputs
        self.opr_maps = {}  # {mge_opr.id: id(ir_opr)}

        self.irgraph = IRGraph()
        self.resolver = SymbolVarResolver(self.irgraph)

    def resolve(self):
        for mge_opr in self.ori_all_oprs:
            if get_opr_type(mge_opr) == "Host2DeviceCopy":
                # add graph inputs
                for inp_var in mge_opr.outputs:
                    self.irgraph.add_net_inputs(self.resolver.get_ir_tensor(inp_var))
                continue
            self.add_opr(mge_opr)

        # add graph outputs
        for out_var in self._orig_outputs:
            self.irgraph.add_net_outputs(self.resolver.get_ir_tensor(out_var))

        return self.irgraph

    def add_opr(self, mge_opr):
        assert mge_opr.id not in self.opr_maps.keys()

        op_gen_cls = MGE2OP.get(get_opr_type(mge_opr), None)
        assert op_gen_cls, "OP {} is not supported".format(get_opr_type(mge_opr))
        ir_op = op_gen_cls(mge_opr, self.irgraph).get_opr()

        self.irgraph.add_op(ir_op)
        self.opr_maps[mge_opr.id] = id(ir_op)
