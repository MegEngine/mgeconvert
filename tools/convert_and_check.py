import argparse
import io
import time
from collections import OrderedDict

import caffe
import megengine.core.tensor.megbrain_graph as G
import megengine.utils.comp_graph_tools as cgtools
import numpy as np
import onnxruntime as ort
from megengine.core._imperative_rt import make_h2d
from megengine.utils.comp_graph_tools import GraphInference
from mgeconvert.caffe_converter.caffe_converter import CaffeConverter
from mgeconvert.mge_context import TopologyNetwork
from mgeconvert.onnx_converter.onnx_converter import OnnxConverter


def change_batch_and_dump(inp_file, oup_file):
    cg, _, outputs = G.load_graph(inp_file)
    inputs = cgtools.get_dep_vars(outputs[0], "Host2DeviceCopy")
    replace_dict = {}
    for var in inputs:
        n_shape = list(var.shape)
        n_shape[0] = 1
        new_input = make_h2d(cg, "xpux", var.dtype, n_shape, var.name)
        replace_dict[var] = new_input

    new_outputs = cgtools.replace_vars(outputs, replace_dict)
    dump_content, _ = G.dump_graph(map(G.VarNode, new_outputs), keep_var_name=2)
    if isinstance(oup_file, str):
        with open(oup_file, "wb") as file:
            file.write(dump_content)
    else:
        oup_file.write(dump_content)


def check_caffe_result(net, inputs, mge_results, prototxt, caffemodel):
    # convert to caffe
    converter = CaffeConverter(net)
    converter.convert()
    assert isinstance(prototxt, str) and isinstance(
        caffemodel, str
    ), "'prototxt' and 'caffemodel' must be string"
    converter.dump(prototxt, caffemodel)
    # run caffe
    caffe_net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    for var_name in inputs:
        if var_name in caffe_net.blobs.keys():
            caffe_net.blobs[var_name].data[...] = inputs[var_name]
    caffe_net.forward()

    # check results
    compared = 0
    for var_name in mge_results:
        mge_data = mge_results[var_name]
        if var_name in caffe_net.blobs.keys():
            caffe_data = caffe_net.blobs[var_name].data
            assert caffe_data.shape == mge_data.shape
            np.testing.assert_allclose(caffe_data, mge_data, rtol=1e-5, atol=1e-1)
            compared += 1
    assert compared == len(mge_results)
    print("pass caffe convert and check")


def check_onnx_result(net, inputs, mge_results, oup_file, version=8, graph="graph"):
    converter = OnnxConverter(net, opset_version=version, graph_name=graph)
    model = converter.convert()
    assert isinstance(oup_file, str)
    with open(oup_file, "wb") as fout:
        fout.write(model.SerializeToString())
    onnx_net = ort.InferenceSession(oup_file)

    onnx_inputs = {}
    for var in onnx_net.get_inputs():
        var_name = var.name
        if var_name in inputs:
            onnx_inputs[var_name] = inputs[var_name]
    assert len(onnx_inputs) == len(inputs)
    pred_onx = onnx_net.run(None, onnx_inputs)
    onnx_outputs = [x.name for x in onnx_net.get_outputs()]
    assert len(onnx_outputs) == len(mge_results)
    compared = 0
    for var_name in mge_results:
        mge_data = mge_results[var_name]
        for i, name in enumerate(onnx_outputs):
            if var_name == name:
                onnx_data = pred_onx[i]
                assert onnx_data.shape == mge_data.shape
                assert onnx_data.dtype == mge_data.dtype
                np.testing.assert_allclose(onnx_data, mge_data, rtol=1e-5, atol=1e-1)
                compared += 1
                break
    assert compared == len(onnx_outputs)
    print("pass onnx convert and check")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Input megengine dump model file"
    )
    parser.add_argument(
        "-c", "--prototxt", required=True, type=str, help="Output caffe .prototxt file"
    )
    parser.add_argument(
        "-b",
        "--caffemodel",
        required=True,
        type=str,
        help="Output caffe .caffemodel file",
    )
    parser.add_argument(
        "-o", "--onnxoutput", required=True, type=str, help="Output onnx .onnx file"
    )
    parser.add_argument("--onnxopset", default=8, type=int, help="Onnx opset version")
    parser.add_argument(
        "--onnxgraph", default="graph", type=str, help="Onnx graph name"
    )

    parser.add_argument(
        "--end_point",
        default=None,
        type=str,
        help="end_point is used to specify which part of the mge model should be converted",
    )

    args = parser.parse_args()
    outspec = None
    if args.end_point is not None:
        outspec = args.end_point.split(";")

    # change batchsize to 1
    input_file = io.BytesIO()
    change_batch_and_dump(args.input, input_file)
    input_file.seek(0)

    inputs = {}
    net = TopologyNetwork(input_file, outspec=outspec)
    for var in net.input_vars:
        shape = list(var.shape)
        data = np.random.randint(0, high=255, size=shape, dtype=np.uint8)
        data = data.astype(var.dtype)
        inputs[var.name] = data

    # inference mge
    input_file.seek(0)
    inference = GraphInference(input_file)
    mge_outputs = inference.run(inp_dict=inputs)
    mge_results = OrderedDict()
    for var_name, value in mge_outputs.items():
        var_name = var_name.replace(":", "_")
        var_name = var_name.replace(".", "_")
        var_name = var_name.replace(",", "_")
        mge_results[var_name] = value

    check_caffe_result(net, inputs, mge_results, args.prototxt, args.caffemodel)
    time.sleep(1)
    check_onnx_result(
        net, inputs, mge_results, args.onnxoutput, args.onnxopset, args.onnxgraph
    )


if __name__ == "__main__":
    main()
