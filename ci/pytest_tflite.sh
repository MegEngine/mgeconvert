#!/bin/bash -e

set -e

./mgeconvert/tflite_converter/init.sh

# try to find libflatbuffers.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

sudo python3 -m pip uninstall flatbuffers -y
sudo python3 -m pip install tensorflow==2.4.0

sudo -H python3 -m pip install -q megengine==1.3.1 -f https://megengine.org.cn/whl/mge.html
pytest test/test_tflite.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.2.0 -f https://megengine.org.cn/whl/mge.html
pytest -v test/test_tflite.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.1.0 -f https://megengine.org.cn/whl/mge.html
pytest -v test/test_tflite.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.0.0 -f https://megengine.org.cn/whl/mge.html
pytest -v test/test_tflite.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==0.6.0 -f https://megengine.org.cn/whl/mge.html
pytest -v test/test_tflite.py
sudo -H python3 -m pip uninstall -y megengine
