# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from google.protobuf import text_format  # type: ignore[attr-defined]

from ..mge_context import TopologyNetwork
from ..mge_context.mge_utils import get_symvar_value
from .caffe_op import MGE2CAFFE
from .caffe_pb import caffe_pb2 as cp  # pylint: disable=import-error


class CaffeConverter:
    def __init__(self, toponet, use_empty_blobs=False):
        self.net = toponet
        self.var2blob_map = {}
        self.layers = []
        self._names = set()
        self._count = 0
        self.use_empty_blobs = use_empty_blobs

    def dump(self, proto_file, caffe_file=None):
        CaffeNet = cp.NetParameter(layer=self.layers)
        if caffe_file is not None:
            with open(caffe_file, "wb") as f:
                f.write(CaffeNet.SerializeToString())

        for layer in CaffeNet.layer:
            layer.ClearField("blobs")

        with open(proto_file, "w") as f:
            f.write(text_format.MessageToString(CaffeNet))

    @property
    def gen_name(self):
        self._count = self._count + 1
        while "_caffe_{0}".format(self._count) in self._names:
            self._count = self._count + 1
        return "_caffe_{0}".format(self._count)

    def get_blob_name(self, varNode):
        if varNode not in self.var2blob_map:
            raise KeyError("can not find VarNode {}".format(varNode))
        return self.var2blob_map[varNode]

    def set_blob_name(self, varNode, name=None):
        assert varNode not in self.var2blob_map, "{} already be set".format(varNode)
        if name is not None:
            assert isinstance(name, str)
            self.var2blob_map[varNode] = name
        else:
            self.var2blob_map[varNode] = self.gen_name
        self._names.add(self.var2blob_map[varNode])
        return self.var2blob_map[varNode]

    def reset_blob_name(self, varNode, name=None):
        assert varNode in self.var2blob_map, "{} should be set".format(varNode)
        if name is not None:
            assert isinstance(name, str)
            self.var2blob_map[varNode] = name
        else:
            self.var2blob_map[varNode] = self.gen_name
        self._names.add(self.var2blob_map[varNode])
        return self.var2blob_map[varNode]

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

        def need_convert(opr):
            is_const = [data.np_data is not None for data in opr.inp_vars]
            return not all(is_const) or len(opr.inp_vars) == 0

        for opr in self.net.all_oprs:
            if not need_convert(opr):
                for tensor in opr.out_vars:
                    if tensor.np_data is None:
                        tensor.np_data = get_symvar_value(tensor._var)
                continue
            MGE2CAFFE[type(opr)](opr, self)


def convert_to_caffe(
    mge_fpath, prototxt="out.prototxt", caffemodel="out.caffemodel", outspec=None
):
    """
    Convert megengine model to Caffe,
    and save caffe model to `prototxt` and `caffemodel`.

    :param mge_fpath: the file path of megengine model.
    :type mge_fpath: str
    :param prototxt: the filename used for saved model definition.
    :type prototxt: str
    :param caffemodel: the filename used for saved model weights.
    :type caffemodel: str
    """

    assert isinstance(mge_fpath, str), "mge_fpath must be string"
    net = TopologyNetwork(mge_fpath, outspec=outspec)
    converter = CaffeConverter(net)
    converter.convert()
    assert isinstance(prototxt, str) and isinstance(
        caffemodel, str
    ), "'prototxt' and 'caffemodel' must be string"
    converter.dump(prototxt, caffemodel)
