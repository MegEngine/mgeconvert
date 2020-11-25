# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from collections import OrderedDict

from .mge_op import str_to_mge_class
from .mge_tensor import Tensor
from .mge_utils import (
    eval_partial,
    get_mge_version,
    get_opr_type,
    get_oprs_seq,
    load_comp_graph_from_file,
)


class TopologyNetwork:
    def __init__(self, model_path):
        _, outputs = load_comp_graph_from_file(model_path)
        all_oprs = get_oprs_seq(outputs)
        self.input_vars = []
        self._orig_inputs = []
        self.output_vars = []
        self._orig_outputs = outputs
        self.all_oprs_map = OrderedDict()
        self.all_vars_map = OrderedDict()

        for mge_opr in all_oprs:
            if get_opr_type(mge_opr) == "Host2DeviceCopy":
                self._orig_inputs.extend(mge_opr.outputs)
            self.add_opr(mge_opr)

        for mge_opr in all_oprs:
            opr = self.get_opr(mge_opr)
            # set inp var
            for x in mge_opr.inputs:
                opr.add_inp_var(self.get_var(x))
            # set out var
            for x in mge_opr.outputs:
                opr.add_out_var(self.get_var(x))
            # set inp, out oprs
            for x in mge_opr.inputs:
                var = self.get_var(x)
                if var.np_data is None:
                    owner = x.owner_opr if get_mge_version() <= "0.6.0" else x.owner
                    inp_opr = self.get_opr(owner)
                    if inp_opr is not None:
                        inp_opr.add_out_opr(opr)
                        opr.add_inp_opr(inp_opr)

        for x in self._orig_inputs:
            self.input_vars.append(self.get_var(x))

        for x in self._orig_outputs:
            self.output_vars.append(self.get_var(x))

    def run(self, feed_input, end_op):
        if end_op is None:
            return eval_partial(feed_input, self.output_vars[-1].sym_var)
        if isinstance(end_op, str):
            for opr in self.all_oprs:
                if opr.name == end_op:
                    end_op = opr
                    break
        assert not isinstance(end_op, str), "end_op op does not exist"
        if not len(end_op.out_vars):
            return []
        flag = False
        for var in end_op.out_vars:
            if var.np_data is None:
                flag = True
                break
        if not flag:
            return [var.np_data for var in end_op.out_vars]
        oup = []
        for var in end_op.out_vars:
            oup.append(var.sym_var)
        # WARN only calculate the first output var
        oup = (oup[0],)
        return eval_partial(feed_input, oup)

    def run_all(self, feed_input):
        result_dict = {}
        for opr in self.all_oprs:
            out = self.run(feed_input, opr)
            result_dict[opr.name] = out
        return result_dict

    def add_opr(self, x):
        assert x.id not in self.all_oprs_map
        self.all_oprs_map[x.id] = str_to_mge_class(get_opr_type(x) + "Opr")(x)

    def get_opr(self, x):
        if x.id in self.all_oprs_map:
            return self.all_oprs_map[x.id]
        else:
            return None

    def get_var(self, x):
        # auto convert to Tensor
        if x.id not in self.all_vars_map:
            owner = x.owner_opr if get_mge_version() <= "0.6.0" else x.owner
            self.all_vars_map[x.id] = Tensor(x, self.get_opr(owner))
        return self.all_vars_map[x.id]

    @property
    def all_oprs(self):
        return self.all_oprs_map.values()

    @property
    def all_vars(self):
        return self.all_vars_map.values()
