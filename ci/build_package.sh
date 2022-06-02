#!/bin/bash -e
basepath=$(cd `dirname $0`; pwd)

. ${basepath}/../mgeconvert/backend/ir_to_tflite/build_flatbuffer.sh

cd $basepath/..
python3 setup.py sdist
python3 setup.py bdist_wheel tflite
python3 setup.py bdist_wheel caffe
python3 setup.py bdist_wheel onnx