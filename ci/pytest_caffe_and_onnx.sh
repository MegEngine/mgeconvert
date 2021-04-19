#!/bin/bash -e

set -e

python3 -m pip install -q -r ci/requires-test.txt

python3 -m pip install --no-binary=protobuf protobuf==3.8.0

apt install -y protobuf-compiler

./mgeconvert/caffe_converter/init.sh

pip3 install scikit-image==0.17.2


sudo -H python3 -m pip install -q megengine==1.3.1 -f https://megengine.org.cn/whl/mge.html
pytest test/test_caffe.py
pytest test/test_onnx.py
sudo -H python3 -m pip uninstall -y megengine


sudo -H python3 -m pip install -q megengine==1.2.0 -f https://megengine.org.cn/whl/mge.html
pytest test/test_caffe.py
pytest test/test_onnx.py
sudo -H python3 -m pip uninstall -y megengine


sudo -H python3 -m pip install -q megengine==1.1.0 -f https://megengine.org.cn/whl/mge.html
pytest test/test_caffe.py
pytest test/test_onnx.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.0.0 -f https://megengine.org.cn/whl/mge.html
pytest test/test_caffe.py
pytest test/test_onnx.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==0.6.0 -f https://megengine.org.cn/whl/mge.html
pytest test/test_caffe.py
pytest test/test_onnx.py
sudo -H python3 -m pip uninstall -y megengine