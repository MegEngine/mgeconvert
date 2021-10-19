# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine as mge
import numpy as np
from megengine import get_logger as mge_get_logger

mge_version = mge.__version__


if mge_version <= "0.6.0":
    # pylint: disable=import-error, no-name-in-module
    import megengine._internal as mgb
    from megengine._internal import cgtools
else:
    import megengine.core._imperative_rt as rt
    import megengine.core.tensor.megbrain_graph as G
    import megengine.utils.comp_graph_tools as cgtools

    if mge_version <= "1.1.0":
        from megengine.core.tensor.raw_tensor import (  # pylint: disable=no-name-in-module,import-error
            as_raw_tensor as Tensor,
        )
    else:
        from megengine.tensor import Tensor


def get_logger(*args):
    return mge_get_logger(*args)


def get_mge_version():
    return mge_version


def get_symvar_value(sym_var):
    if mge_version <= "0.6.0":
        if sym_var.inferred_value is not None:
            val = sym_var.inferred_value
            return val
        else:
            cg = sym_var.owner_graph
            func = cg.compile_outonly(sym_var)
            val = func()
            return val
    else:
        if sym_var.value is not None:
            return sym_var.value
        else:
            out_node = G.ValueOutputNode(sym_var)
            cg = out_node.outputs[0].graph
            func = cg.compile(out_node.outputs)
            func.execute()
            return out_node.get_value()


def isnum(x):
    return isinstance(x, (int, float))


def isconst(x):
    return x.np_data is not None


def isvar(x):
    return (
        isinstance(x, mgb.SymbolVar)
        if mge_version <= "0.6.0"
        else isinstance(x, rt.VarNode)  # pylint: disable=c-extension-no-member
    )


def get_shape(x):
    return x._get_imm_shape() if mge_version <= "0.6.0" else x.shape


def get_dep_vars(x, type=None):
    return cgtools.get_dep_vars(x, type)


def get_dtype_name(x):
    return (
        x.dtype.metadata["mgb_dtype"]["name"] if isinstance(x.dtype, np.dtype) else None
    )


def get_opr_type(x):
    return cgtools.get_opr_type(x)


def get_owner_opr_type(x):
    if mge_version <= "0.6.0":
        return cgtools.get_type(x._var)
    else:
        return cgtools.get_owner_opr_type(x._var)


def load_comp_graph_from_file(path):
    if mge_version <= "0.6.0":
        cg, _, outputs = mgb.load_comp_graph_from_file(path)
    else:
        ret = G.load_graph(path)
        cg = ret.graph
        outputs = ret.output_vars_list
    return cg, outputs


def graph_traversal(outputs):
    (
        map_oprs,
        map_vars,
        var2oprs,
        opr2receivers,
        indegree2opr,
        opr2indegree,
    ) = cgtools.graph_traversal(outputs)
    return map_oprs, map_vars, var2oprs, opr2receivers, indegree2opr, opr2indegree


def get_oprs_seq(outputs, prune_reshape=True):
    all_oprs = cgtools.get_oprs_seq(outputs, prune_reshape=prune_reshape)
    return all_oprs


def eval_partial(inp, oup):
    if not isinstance(oup, (list, tuple)):
        oup = (oup,)
    inputs = cgtools.get_dep_vars(oup, "Host2DeviceCopy")
    if mge_version <= "0.6.0":
        cg = oup[0].owner_graph
        outputs = list(map(mgb.copy_output, oup))
        f = cg.compile(inputs, outputs)
        result = f(inp)
    else:
        if not isinstance(inp, (list, tuple)):
            inp = (inp,)
        replace_dict = {}
        inp_node_list = []
        for i in inputs:
            inp_node = G.InputNode(
                device="xpux", dtype=inputs[0].dtype, graph=inputs[0].graph
            )
            replace_dict[i] = inp_node.outputs[0]
            inp_node_list.append(inp_node)
        new_out = cgtools.replace_vars(oup, replace_dict)
        out_node_list = [G.OutputNode(i) for i in new_out]
        new_out_list = [i.outputs[0] for i in out_node_list]
        cg = new_out_list[0].graph
        func = cg.compile(new_out_list)
        for node, value in zip(inp_node_list, inp):
            node.set_value(Tensor(value)._dev_tensor())
        func.execute()
        result = [o.get_value().numpy() for o in out_node_list]

    return result
