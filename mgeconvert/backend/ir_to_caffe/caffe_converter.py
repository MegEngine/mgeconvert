# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os
from typing import Dict, List, Sequence, Set, Union

# pylint: disable=import-error
from google.protobuf import text_format  # type: ignore
from megengine import get_logger
from tabulate import tabulate
from tqdm import tqdm

from ...converter_ir.ir_graph import IRGraph, OpBase
from ...converter_ir.ir_quantizer import IRQuantizer
from ...converter_ir.ir_tensor import IRTensor  # pylint: disable=unused-import
from .caffe_op import MGE2CAFFE, BackEnd, _add_input_layer

logger = get_logger(__name__)

if "USE_CAFFE_PROTO" not in os.environ:
    from .caffe_pb import caffe_pb2 as cp
else:
    from caffe.proto import caffe_pb2 as cp


class ReorderError(Exception):
    pass


def _get_dep_opr(tensors: Union[Sequence[IRTensor], IRTensor]) -> List[OpBase]:
    if not isinstance(tensors, Sequence):
        tensors = (tensors,)
    ret = []  # type: List[OpBase]
    queue = list(tensors)  # type: List[IRTensor]
    visited_queue = []  # type: List[IRTensor]
    while queue:
        tensor = queue.pop()
        visited_queue.append(tensor)

        opr = tensor.owner_opr

        if not isinstance(opr, OpBase):
            continue

        if opr not in ret:
            ret.append(opr)

        for i in opr.inp_tensors:
            if i not in queue and i not in visited_queue:
                queue.append(i)
    return ret


def _check_dependency(outputs: List[IRTensor]):
    """
    Check whether there exist one output depend on another output
    """
    output_oprs = {var.owner_opr for var in outputs}
    output_input_oprs = {
        i.owner_opr for opr in _get_dep_opr(outputs) for i in opr.inp_tensors
    }
    if len(output_oprs) != len(outputs) or (output_oprs & output_input_oprs):
        raise ReorderError("Bad order due to dependency between two outputs.")


def _reorder_outputs(context: "CaffeConverter"):
    """
    Try to keep same order with original network, but sometimes it is impossible,
    raise a ReorderError if it can't order output layers correctly.
    """
    output_tensor = context.net.graph_outputs
    _check_dependency(output_tensor)

    blob2index = {}
    ordered_layers = []
    output_layers = [None] * len(output_tensor)

    for i, oup in enumerate(output_tensor):
        blob = context.get_blob_name(oup)
        blob2index[blob] = i

    for l in context.layers:
        is_output = False
        for blob in l.top:
            idx = blob2index.get(blob, None)
            if idx is not None:
                if not is_output:
                    is_output = True
                else:
                    raise ReorderError(
                        "layer {} has more than one network outputs".format(l)
                    )
                if output_layers[idx] is not None:
                    raise ReorderError(
                        "duplicated blob name of layer {} and {}".format(
                            output_layers[idx], l
                        )
                    )
                output_layers[idx] = l

        if not is_output:
            ordered_layers.append(l)

    if output_layers.count(None) > 0:
        raise ReorderError("failure to replace all output vars.")

    ordered_layers += output_layers
    return ordered_layers


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

    def update_quantize_dict(self, tensor, name=None):
        if tensor.q_dtype is not None:
            tname = name if name is not None else tensor.name
            self.quantizer.set_quant_info(tname, tensor)

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

        try:
            layers = _reorder_outputs(self)
            self.layers = layers
        except ReorderError as e:
            logger.warning(str(e))
            logger.warning(
                "Can not keep the same order with original network, "
                "ignore reorder error and fallback to unordered caffe network."
            )
            header = ["megengine output tensor", "caffe blob name"]
            outputs = []
            for var in self.net.graph_outputs:
                blob = self.get_blob_name(var)
                outputs.append([var.name, blob])
            logger.info(tabulate(outputs, header))
