# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import List  # pylint: disable=unused-import
from typing import Sequence

from megengine.logger import get_logger

from .ir_op import OpBase
from .ir_tensor import IRTensor

logger = get_logger(__name__)


class IRGraph:
    def __init__(self) -> None:
        self.graph_inputs = []  # type: List[IRTensor]
        self.graph_outputs = []  # type: List[IRTensor]

        self._tensor_ids = []  # type: List[int]
        self._opr_ids = []  # type: List[int]
        self.all_tensors = []  # type: List[IRTensor]
        self.all_oprs = []  # type: List[OpBase]

    def add_op(self, op, index=None):
        assert len(self._opr_ids) == len(self.all_oprs)
        if index is not None:
            self._opr_ids.insert(index, id(op))
            self.all_oprs.insert(index, op)
        else:
            if not isinstance(op, Sequence):
                ops = (op,)
            for opr in ops:
                self.all_oprs.append(opr)
                self._opr_ids.append(id(opr))

    def delete_ops(self, index):
        del self.all_oprs[index]
        del self._opr_ids[index]

    def replace_op(self, old_op, new_op):
        index = self.all_oprs.index(old_op)
        self.all_oprs[index] = new_op
        self._opr_ids[index] = id(new_op)

    def get_tensor(self, var_id, ir_tensor: IRTensor, origin_tensor: IRTensor = None):
        """
        var_id: origin var id
        origin_tensor: relate the var to a already existed ir tensor
        """
        if var_id:
            if var_id not in self._tensor_ids:
                self._tensor_ids.append(var_id)
                if origin_tensor:
                    self.all_tensors.append(origin_tensor)
                else:
                    self.all_tensors.append(ir_tensor)
            return self.all_tensors[self._tensor_ids.index(var_id)]
        else:  # int, float
            return ir_tensor

    def add_tensor(self, var_id, ir_tensor):
        if var_id not in self._tensor_ids:
            self._tensor_ids.append(var_id)
            self.all_tensors.append(ir_tensor)

    def find_inp_oprs(self, op):
        if len(op.inp_tensors) == 0:
            return None
        inp_oprs = []
        for inp in op.inp_tensors:
            if inp.owner_opr is None:
                continue
            inp_oprs.append(inp.owner_opr)
        return inp_oprs

    def find_out_oprs(self, op):
        out_oprs = []
        for oup in op.out_tensors:
            if oup.user_opr is not None:
                out_oprs.extend(oup.user_opr)
        return out_oprs

    def add_net_inputs(self, inp_tensor: IRTensor):
        self.graph_inputs.append(inp_tensor)
        inp_tensor.owner_opr = self

    def add_net_outputs(self, out_tensor: IRTensor):
        self.graph_outputs.append(out_tensor)

    def insert_op_after(self, new_op: OpBase, last_op: OpBase):
        """
        only consider cases of one output
        """
        out_oprs = self.find_out_oprs(last_op)
        if len(out_oprs) == 0:
            # last op of the graph
            for t in new_op.inp_tensors:
                t.user_opr.append(new_op)
                if t in self.graph_outputs:
                    self.graph_outputs.remove(t)
            for t in new_op.out_tensors:
                t.owner_opr = new_op
                self.graph_outputs.append(t)
        else:
            assert len(out_oprs) == 1, "Do not support more than one output oprs yet."
            next_op = out_oprs[0]
            for t in new_op.inp_tensors:
                if t.owner_opr == last_op:
                    assert next_op in t.user_opr
                    idx = t.user_opr.index(next_op)
                    t.user_opr[idx] = new_op
            for t in new_op.out_tensors:
                t.owner_opr = new_op
                assert t in next_op.inp_tensors
                t.user_opr.append(next_op)
        op_idx = self.all_oprs.index(last_op) + 1
        self.add_op(new_op, op_idx)

    def insert_op_before(self, new_op: OpBase, next_op: OpBase):
        inp_oprs = self.find_inp_oprs(next_op)
        if len(inp_oprs) == 0:
            # first op of the graph
            for t in new_op.inp_tensors:
                assert next_op in t.user_opr
                idx = t.user_opr.index(next_op)
                t.user_opr[idx] = new_op
                if t not in self.graph_inputs:
                    self.graph_inputs.append(t)
                    logger.warning("WARNING: Add graph inputs(%s)", t.name)
            for t in new_op.out_tensors:
                t.owner_opr = new_op
        else:
            assert len(inp_oprs) == 1, "Do not support more than one input oprs yet."
            last_op = inp_oprs[0]
            for t in new_op.inp_tensors:
                if t.owner_opr == last_op:
                    assert next_op in t.user_opr
                    idx = t.user_opr.index(next_op)
                    t.user_opr[idx] = new_op
            for t in new_op.out_tensors:
                t.owner_opr = new_op
                assert t in next_op.inp_tensors
                t.user_opr.append(next_op)
        op_idx = self.all_oprs.index(next_op)
        self.add_op(new_op, op_idx)

    def __repr__(self):
        res = ""
        for op in self.all_oprs:
            res += op.name + "\n"
            res += "\tInput:\n"
            for i in op.inp_tensors:
                res += "\t\t{}\n".format(i.name)
            res += "\tOutput:\n"
            for i in op.out_tensors:
                res += "\t\t{}\n".format(i.name)
        return res
