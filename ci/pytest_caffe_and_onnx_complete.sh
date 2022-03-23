#!/bin/bash -e

set -e

python3 -m pip install -q -r ci/requires-test.txt

python3 -m pip install --no-binary=protobuf protobuf==3.11.1

apt install -y protobuf-compiler

./mgeconvert/backend/ir_to_caffe/init.sh

pip3 install scikit-image==0.17.2 onnx onnxruntime

export USE_CAFFE_PROTO=1

function version_gt() { test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" != "$1"; }

MGE_VERSIONS="1.9.0 1.8.1 1.7.0 1.6.0 1.5.0 1.4.0 1.3.0 1.2.0 1.1.0 1.0.0"
for VER in $MGE_VERSIONS
do
    sudo -H python3 -m pip install -q megengine==$VER -f https://megengine.org.cn/whl/mge.html
    pytest test/mge/test_caffe.py
    pytest test/mge/test_onnx.py
    if version_gt $VER "1.5.0";then
        pytest test/traced_module/test_caffe.py
        pytest test/traced_module/test_onnx.py
    fi
    sudo -H python3 -m pip uninstall -y megengine
done

unset USE_CAFFE_PROTO
