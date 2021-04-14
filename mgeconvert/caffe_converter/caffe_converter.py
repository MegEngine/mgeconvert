# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
from google.protobuf import text_format  # type: ignore[attr-defined]
from tqdm import tqdm

from ..mge_context import TopologyNetwork
from ..mge_context import mge_op as Op
from ..mge_context.mge_utils import get_symvar_value, isconst
from .caffe_op import MGE2CAFFE
from .caffe_pb import caffe_pb2 as cp  # pylint: disable=import-error


class Node:
    def __init__(self, type, is_output=False, const_value=None):
        self.opnode = None
        self.type = type
        self.inp_oprs = []
        self.inp_const = []
        self.inp_vars = []
        self.is_output = is_output
        self.const_value = const_value

    def check_const_value(self, opnode):
        inp_vars = [v.np_data for v in opnode.inp_vars]
        for const in self.const_value:
            idx = const[0]
            if idx == -1:
                find = False
                for index, v in enumerate(inp_vars):
                    if np.array_equal(const[1], v):
                        find = True
                        del inp_vars[index]
                        break
                if not find:
                    return False
            elif not np.array_equal(const[1], inp_vars[idx]):
                return False
        return True


def get_type(opr):
    if isinstance(opr, Op.ElemwiseOpr):
        return opr.mode
    else:
        return str(type(opr))


def match(node, opr):
    node_queue = [node]
    opr_queue = [opr]
    matched_opr = set()
    matched_node = set()
    while len(node_queue) != 0:
        cur_node = node_queue.pop(0)
        cur_opr = opr_queue.pop(0)
        if cur_node.type != get_type(cur_opr) and cur_node.type != "*" or cur_opr.skip:
            return False
        if cur_node.opnode == None:
            cur_node.opnode = cur_opr
            if cur_node.const_value != None:
                if not cur_node.check_const_value(cur_opr):
                    return False
        elif cur_node.opnode != cur_opr:
            return False

        matched_opr.add(cur_opr)
        matched_node.add(cur_node)
        for i, var in enumerate(cur_opr.inp_vars):
            if isconst(var):
                cur_node.inp_const.append([i, var.np_data])
            else:
                cur_node.inp_vars.append([i, var])
        if len(cur_node.inp_oprs) == 0:
            continue
        if len(cur_node.inp_oprs) != len(cur_opr.inp_oprs):
            return False

        for i, j in zip(cur_node.inp_oprs, cur_opr.inp_oprs):
            node_queue.append(i)
            opr_queue.append(j)

    for n in matched_node:
        if n.is_output:
            continue
        for op in n.opnode.out_oprs:
            if op not in matched_opr:
                return False

    return True


def _leaky_relu(opr):

    AddNode = Node("ADD", is_output=True)
    MulNode = Node("MUL")
    MaxNode = Node("MAX", const_value=[(-1, [0.0])])
    MinNode = Node("MIN", const_value=[(-1, [0.0])])
    AddNode.inp_oprs = [MaxNode, MulNode]
    MulNode.inp_oprs = [MinNode]
    if (
        len(opr.out_oprs) == 1
        and isinstance(opr.out_oprs[0], Op.ElemwiseOpr)
        and opr.out_oprs[0].mode == "ADD"
    ):
        add_opr = opr.out_oprs[0]
        if match(AddNode, add_opr):
            if (
                MaxNode.inp_vars[0] == MinNode.inp_vars[0]
                and len(MulNode.inp_const) == 1
                and MulNode.inp_const[0][1].shape == (1,)
            ):
                LeakRelu = Op.LeakyReluOpr(
                    "LeakRelu_" + AddNode.opnode.name, MulNode.inp_const[0][1],
                )
                LeakRelu.inp_vars = [MaxNode.inp_vars[0][1]]
                LeakRelu.out_vars = AddNode.opnode.out_vars
                AddNode.opnode.skip = True
                MulNode.opnode.skip = True
                MaxNode.opnode.skip = True
                MinNode.opnode.skip = True
                return LeakRelu
    return None


def _conv_bias(opr):

    BiasNode = Node("ADD", is_output=True)
    ConvNode = Node(str(Op.ConvolutionForwardOpr))
    BiasNode.inp_oprs = [ConvNode]
    if (
        len(opr.out_oprs) == 1
        and isinstance(opr.out_oprs[0], Op.ElemwiseOpr)
        and opr.out_oprs[0].mode == "ADD"
    ):
        add_opr = opr.out_oprs[0]
        if match(BiasNode, add_opr):
            ConvForwardBias = Op.ConvForwardBiasOpr(
                "ConvForwardBias_" + BiasNode.opnode.name,
                ConvNode.opnode._opr,
                BiasNode.inp_const[0][1],
            )
            ConvForwardBias.inp_vars = ConvNode.opnode.inp_vars
            ConvForwardBias.out_vars = BiasNode.opnode.out_vars
            BiasNode.opnode.skip = True
            ConvNode.opnode.skip = True
            return ConvForwardBias
    return None


def _deconv_bias(opr):

    BiasNode = Node("ADD", is_output=True)
    ConvNode = Node(str(Op.ConvolutionBackwardDataOpr))
    BiasNode.inp_oprs = [ConvNode]
    if (
        len(opr.out_oprs) == 1
        and isinstance(opr.out_oprs[0], Op.ElemwiseOpr)
        and opr.out_oprs[0].mode == "ADD"
    ):
        add_opr = opr.out_oprs[0]
        if match(BiasNode, add_opr):
            ConvolutionBackwardDataBias = Op.ConvolutionBackwardDataBiasOpr(
                "ConvolutionBackwardDataBias_" + BiasNode.opnode.name,
                ConvNode.opnode._opr,
                BiasNode.inp_const[0][1],
            )
            ConvolutionBackwardDataBias.inp_vars = ConvNode.opnode.inp_vars
            ConvolutionBackwardDataBias.out_vars = BiasNode.opnode.out_vars
            BiasNode.opnode.skip = True
            ConvNode.opnode.skip = True
            return ConvolutionBackwardDataBias
    return None


def _fully_connected(opr):

    BiasNode = Node("ADD", is_output=True)
    MMNode = Node(str(Op.MatrixMulOpr))
    BiasNode.inp_oprs = [MMNode]
    if (
        len(opr.out_oprs) == 1
        and isinstance(opr.out_oprs[0], Op.ElemwiseOpr)
        and opr.out_oprs[0].mode == "ADD"
    ):
        add_opr = opr.out_oprs[0]
        if match(BiasNode, add_opr):
            FullyConnected = Op.FullyConnectedOpr(
                "FullyConnected_" + BiasNode.opnode.name,
                MMNode.opnode._opr,
                BiasNode.inp_const[0][1],
            )
            FullyConnected.inp_vars = MMNode.opnode.inp_vars
            FullyConnected.out_vars = BiasNode.opnode.out_vars
            BiasNode.opnode.skip = True
            MMNode.opnode.skip = True
            return FullyConnected
    return None


replace_rules = {}
replace_rules["MAX"] = _leaky_relu
replace_rules[str(Op.ConvolutionForwardOpr)] = _conv_bias
replace_rules[str(Op.ConvolutionBackwardDataOpr)] = _deconv_bias
replace_rules[str(Op.MatrixMulOpr)] = _fully_connected


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

        all_oprs = list(self.net.all_oprs)
        for idx, opr in enumerate(all_oprs):
            if get_type(opr) in replace_rules:
                new_op = replace_rules[get_type(opr)](opr)
                if new_op is not None:
                    all_oprs[idx] = new_op

        for index in range(len(all_oprs) - 1, -1, -1):
            if all_oprs[index].skip:
                del all_oprs[index]

        for opr in tqdm(all_oprs):
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
