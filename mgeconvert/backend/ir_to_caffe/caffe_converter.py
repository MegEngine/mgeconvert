# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os
from typing import Dict, List, Set

# pylint: disable=import-error
from google.protobuf import text_format  # type: ignore
from tqdm import tqdm

from ...converter_ir.ir_graph import IRGraph
from ...converter_ir.ir_quantizer import IRQuantizer
from ...converter_ir.ir_tensor import IRTensor  # pylint: disable=unused-import
from .caffe_op import MGE2CAFFE, BackEnd, _add_input_layer

if "USE_CAFFE_PROTO" not in os.environ:
    from .caffe_pb import caffe_pb2 as cp
else:
    from caffe.proto import caffe_pb2 as cp


class CaffeConverter:
    def __init__(
        self,
        net,
        quantizer: IRQuantizer,
        use_empty_blobs=False,
        convert_backend=BackEnd.CAFFE,
    ):
        assert isinstance(net, IRGraph), "net must be instance of IRGraph"
        self.net = net
        self.tensor2blob_map = {}  # type: Dict[IRTensor, str]
        self.layers = []  # type: List
        self._names = set()  # type: Set
        self._count = 0
        self.use_empty_blobs = use_empty_blobs
        self.quantizer = quantizer
        self.convert_backend = convert_backend

    def update_quantize_dict(self, tensor):
        self.quantizer.parse_quant_info(tensor)

    def dump(self, proto_file, caffe_file=None):
        CaffeNet = cp.NetParameter(layer=self.layers)
        if caffe_file is not None:
            with open(caffe_file, "wb") as f:
                f.write(CaffeNet.SerializeToString())

        for layer in CaffeNet.layer:
            layer.ClearField("blobs")

        with open(proto_file, "w", encoding=None) as f:
            f.write(text_format.MessageToString(CaffeNet))

    @property
    def gen_name(self):
        self._count = self._count + 1
        while "_caffe_{0}".format(self._count) in self._names:
            self._count = self._count + 1
        return "_caffe_{0}".format(self._count)

    def get_blob_name(self, tensor):
        if tensor not in self.tensor2blob_map:
            raise KeyError("can not find tensor {}".format(tensor))
        return self.tensor2blob_map[tensor]

    def set_blob_name(self, tensor, name=None):
        assert tensor not in self.tensor2blob_map, "{} already be set".format(tensor)
        if name is not None:
            assert isinstance(name, str)
            self.tensor2blob_map[tensor] = name
        else:
            self.tensor2blob_map[tensor] = self.gen_name
        self._names.add(self.tensor2blob_map[tensor])
        return self.tensor2blob_map[tensor]

    def reset_blob_name(self, tensor, name=None):
        assert tensor in self.tensor2blob_map, "{} should be set".format(tensor)
        if name is not None:
            assert isinstance(name, str)
            self.tensor2blob_map[tensor] = name
        else:
            self.tensor2blob_map[tensor] = self.gen_name
        self._names.add(self.tensor2blob_map[tensor])
        return self.tensor2blob_map[tensor]

    def gen_blob_proto(self, data):
        if self.use_empty_blobs:
            return cp.BlobProto()
        if isinstance(data, (int, float)):
            return cp.BlobProto(data=[data])
        else:
            return cp.BlobProto(
                data=data.reshape(-1), shape=cp.BlobShape(dim=data.shape)
            )

    def add_layer(self, layer):
        if isinstance(layer, list):
            for x in layer:
                self.layers.append(x)
        else:
            self.layers.append(layer)

    def convert(self):
        unsupported_oprs = []
        for opr in self.net.all_oprs:
            if not isinstance(opr, tuple(MGE2CAFFE.keys())):
                unsupported_oprs.append(opr)
                continue
        unsupported_oprs = set(map(type, unsupported_oprs))
        assert not unsupported_oprs, "Operators {} are not supported yet".format(
            unsupported_oprs
        )

        def need_convert(mge_opr):
            is_const = [data.np_data is not None for data in mge_opr.inp_tensors]
            return not all(is_const) and len(mge_opr.inp_tensors) > 0

        all_oprs = list(self.net.all_oprs)

        # add net input
        for net_inp in self.net.graph_inputs:
            _add_input_layer(net_inp, self)

        for index in range(len(all_oprs) - 1, -1, -1):
            if all_oprs[index].skip:
                del all_oprs[index]

        for opr in tqdm(all_oprs):
            if not need_convert(opr):
                continue
            MGE2CAFFE[type(opr)](opr, self)
