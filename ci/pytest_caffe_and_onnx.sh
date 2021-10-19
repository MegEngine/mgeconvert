#!/bin/bash -e

set -e

python3 -m pip install -q -r ci/requires-test.txt

python3 -m pip install --no-binary=protobuf protobuf==3.8.0

apt install -y protobuf-compiler

./mgeconvert/backend/ir_to_caffe/init.sh

pip3 install scikit-image==0.17.2

sudo -H python3 -m pip install -q megengine==1.6.0 -f https://megengine.org.cn/whl/mge.html
pytest test/mge/test_caffe.py
pytest test/mge/test_onnx.py
pytest test/traced_module/test_caffe.py
pytest test/traced_module/test_onnx.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.5.0 -f https://megengine.org.cn/whl/mge.html
pytest test/mge/test_caffe.py
pytest test/mge/test_onnx.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.4.0 -f https://megengine.org.cn/whl/mge.html
pytest test/mge/test_caffe.py
pytest test/mge/test_onnx.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.3.0 -f https://megengine.org.cn/whl/mge.html
pytest test/mge/test_caffe.py
pytest test/mge/test_onnx.py
sudo -H python3 -m pip uninstall -y megengine


sudo -H python3 -m pip install -q megengine==1.2.0 -f https://megengine.org.cn/whl/mge.html
pytest test/mge/test_caffe.py
pytest test/mge/test_onnx.py
sudo -H python3 -m pip uninstall -y megengine


sudo -H python3 -m pip install -q megengine==1.1.0 -f https://megengine.org.cn/whl/mge.html
pytest test/mge/test_caffe.py
pytest test/mge/test_onnx.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.0.0 -f https://megengine.org.cn/whl/mge.html
pytest test/mge/test_caffe.py
pytest test/mge/test_onnx.py
sudo -H python3 -m pip uninstall -y megengine