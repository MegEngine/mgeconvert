import argparse
import csv
from typing import Sequence

import megengine as mge
import numpy as np
from basecls_model import generate_cls_model, get_public_cls_models
from megengine.traced_module import trace_module
from mgeconvert.converters.tm_to_caffe import tracedmodule_to_caffe
from mgeconvert.converters.tm_to_onnx import tracedmodule_to_onnx
from mgeconvert.converters.tm_to_tflite import tracedmodule_to_tflite

max_error = 1e-6
tmp_file = "test_module"


def calculate_error(result, gt):
    absolute_error = np.abs(result - gt)
    average_absolute_error = np.average(absolute_error)
    relative_error = np.abs(result - gt) / np.abs(gt)
    average_relative_error = np.average(relative_error)
    return average_absolute_error, average_relative_error


def get_traced_module(net, *x):
    traced_module = trace_module(net, *x)
    expect = traced_module(*x)
    return traced_module, expect


def _test_convert_caffe_result(inputs, traced_module, mge_results, input_name="x"):
    import caffe  # pylint: disable=import-error

    tracedmodule_to_caffe(
        traced_module, prototxt=tmp_file + ".txt", caffemodel=tmp_file + ".caffemodel"
    )
    caffe_net = caffe.Net(tmp_file + ".txt", tmp_file + ".caffemodel", caffe.TEST)
    for i in caffe_net.blobs.keys():
        if isinstance(input_name, list):
            for idx, name in enumerate(input_name):
                if name in i:
                    caffe_net.blobs[i].data[...] = inputs[idx]
                    break
        else:
            if input_name in i:
                caffe_net.blobs[i].data[...] = inputs
                break
    out_dict = caffe_net.forward()

    if isinstance(mge_results, dict):
        assert len(list(out_dict.keys())) == len(list(mge_results.keys()))
        for name in mge_results.keys():
            assert name._name in out_dict.keys()
            assert out_dict[name._name].shape == mge_results[name].shape
            return calculate_error(out_dict[name._name], mge_results[name])
    else:
        caffe_results = list(out_dict.values())[0]
        assert caffe_results.shape == mge_results.shape
        return calculate_error(caffe_results, mge_results)


def _test_convert_onnx_result(
    inputs, traced_module, mge_result, min_version=7, max_version=12
):
    import onnxruntime as ort

    for version in range(min_version, max_version + 1):
        tracedmodule_to_onnx(
            traced_module, tmp_file + ".onnx", opset=version, graph_name="graph"
        )
        onnx_net = ort.InferenceSession(tmp_file + ".onnx")
        if isinstance(inputs, (list, tuple)):
            input_dict = {}
            for i, inp in enumerate(inputs):
                input_name = onnx_net.get_inputs()[i].name
                X_test = inp
                input_dict[input_name] = X_test
            pred_onx = onnx_net.run(None, input_dict)[0]
        else:
            input_name = onnx_net.get_inputs()[0].name
            X_test = inputs
            pred_onx = onnx_net.run(None, {input_name: X_test})[0]
        assert pred_onx.shape == mge_result.shape
        assert pred_onx.dtype == mge_result.dtype
        return calculate_error(pred_onx, mge_result)


def _test_convert_tflite_result(
    inputs,
    tm,
    tm_result,
    nhwc=True,
    nhwc2=True,
    scale=1,
    zero_point=0,
    require_quantize=False,
):
    from tensorflow.lite.python import interpreter  # pylint: disable=import-error

    if not isinstance(inputs, Sequence):
        inputs = [
            inputs,
        ]
    if not isinstance(scale, Sequence):
        scale = (scale,)
    if not isinstance(zero_point, Sequence):
        zero_point = (zero_point,)
    for i, inp in enumerate(inputs):
        if nhwc and inp.ndim == 4:
            inputs[i] = inp.transpose((0, 2, 3, 1))

    tracedmodule_to_tflite(
        tm, output=tmp_file + ".tflite", require_quantize=require_quantize
    )

    tfl_model = interpreter.Interpreter(model_path=tmp_file + ".tflite")
    tfl_model.allocate_tensors()

    input_details = tfl_model.get_input_details()
    for i, inp in enumerate(inputs):
        tfl_model.set_tensor(input_details[i]["index"], inp)
    tfl_model.invoke()

    pred_tfl = []
    if not isinstance(scale, Sequence):
        scale = (scale,)
        zero_point = (zero_point,)
    for index, i in enumerate(tfl_model.get_output_details()):
        out = tfl_model.tensor(i["index"])()
        if nhwc2 and out.ndim == 4:
            out = out.transpose((0, 3, 1, 2))
        index = len(scale) - 1 if index >= len(scale) else index
        out = ((out - float(zero_point[index])) * scale[index]).astype("float32")
        pred_tfl.append(out)

    if not isinstance(tm_result, Sequence):
        tm_result = (tm_result,)
    absolute_error = []
    relative_error = []

    for i, j, s in zip(tm_result, pred_tfl, scale):
        assert i.shape == j.shape
        assert i.dtype == j.dtype
        a, r = calculate_error(i, j)
        absolute_error.append(a)
        relative_error.append(r)
    return np.average(absolute_error), np.average(relative_error)


def test_cls_models(model_name, framework):
    model = generate_cls_model(model_name)
    model.eval()
    shapes = (1, 3, 224, 224)
    if model.__class__.__name__ == "ViT":
        shapes = (1, 3, *model.patch_embed.img_size)
    inputs = mge.tensor(
        np.random.random(shapes).astype(np.float32)
        # np.random.randint(0, high=1, size=shapes).astype(np.float32)
    )
    print("testing model {}".format(type(model)))
    tm_module, mge_result = get_traced_module(model, mge.tensor(inputs))
    if framework == "caffe":
        return _test_convert_caffe_result(inputs.numpy(), tm_module, mge_result)
    elif framework == "onnx":
        return _test_convert_onnx_result(inputs.numpy(), tm_module, mge_result)
    elif framework == "tflite":
        return _test_convert_tflite_result(inputs.numpy(), tm_module, mge_result)
    else:
        assert False, "doesn't support framework: {}".format(framework)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test mgeconvert")
    parser.add_argument("model_name", type=str, help="model name")
    parser.add_argument("framework", type=str, help="target framework")
    parser.add_argument("output", type=str, help="result save file path")

    args = parser.parse_args()
    try:
        result = test_cls_models(args.model_name, args.framework)
    except Exception:
        result = ("None", "None")

    with open(args.output, "a+") as f:
        csv_writer = csv.writer(f)
        data_raw = [args.model_name, args.framework, result[0], result[1]]
        csv_writer.writerow(data_raw)
