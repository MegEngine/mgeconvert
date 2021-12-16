#!/bin/bash -e

set -e

python3 -m pip install -q -r ci/requires-test.txt

apt install -y protobuf-compiler

pip3 install . --user --install-option="--targets=all"

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH:

sudo python3 -m pip uninstall flatbuffers -y
sudo python3 -m pip install tensorflow==2.5.0
sudo -H python3 -m pip install -q megengine==1.6.0 -f https://megengine.org.cn/whl/mge.html
pip3 install scikit-image==0.17.2


python3 test/gen_models.py

python3 bin/convert tracedmodule_to_tflite -i float_model.tm -o out.tflite
python3 bin/convert mge_to_tflite -i float_model.mge -o out.tflite

python3 bin/convert tracedmodule_to_tflite -i qat_model.tm -o out.tflite --require_quantize
python3 bin/convert tracedmodule_to_tflite -i qat_model.tm -o out.tflite --quantize_file_path quant_params.json
python3 bin/convert tracedmodule_to_tflite -i qat_model.tm -o out.tflite --quantize_file_path quant_params.json --param_fake_quant
python3 bin/convert tracedmodule_to_tflite -i qat_model.tm -o out.tflite --input_data_type quint8 --input_scales 0.125 --input_zero_points 128 --require_quantize


export USE_CAFFE_PROTO=1
python3 bin/convert tracedmodule_to_caffe -i float_model.tm -c out.prototxt -b out.caffemodel
python3 bin/convert mge_to_caffe -i float_model.mge -c out.prototxt -b out.caffemodel

python3 bin/convert tracedmodule_to_caffe -i qat_model.tm -c out.prototxt -b out.caffemodel --quantize_file_path quant_params.json
python3 bin/convert tracedmodule_to_caffe -i qat_model.tm -c out.prototxt -b out.caffemodel --quantize_file_path quant_params.json --param_fake_quant
python3 bin/convert tracedmodule_to_caffe -i qat_model.tm -c out.prototxt -b out.caffemodel --quantize_file_path quant_params.json --input_data_type quint8 --input_scales 0.125 --input_zero_points 128


python3 bin/convert tracedmodule_to_onnx -i float_model.tm -o out.onnx
python3 bin/convert mge_to_onnx -i float_model.mge -o out.onnx

python3 bin/convert onnx_to_tracedmodule -i out.onnx -o out.tm
python3 bin/convert onnx_to_mge -i out.onnx -o out.mge