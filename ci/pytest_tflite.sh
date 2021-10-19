#!/bin/bash -e

set -e

./mgeconvert/backend/ir_to_tflite/init.sh

# try to find libflatbuffers.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

sudo python3 -m pip uninstall flatbuffers -y
sudo python3 -m pip install tensorflow==2.5.0

sudo -H python3 -m pip install -q megengine==1.6.0 -f https://megengine.org.cn/whl/mge.html
pytest test/mge/test_tflite.py
pytest test/traced_module/test_tflite.py
pytest test/traced_module/test_qat_tflite.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.5.0 -f https://megengine.org.cn/whl/mge.html
pytest test/mge/test_tflite.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.4.0 -f https://megengine.org.cn/whl/mge.html
pytest test/mge/test_tflite.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.3.0 -f https://megengine.org.cn/whl/mge.html
pytest test/mge/test_tflite.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.2.0 -f https://megengine.org.cn/whl/mge.html
pytest -v test/mge/test_tflite.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.1.0 -f https://megengine.org.cn/whl/mge.html
pytest -v test/mge/test_tflite.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.0.0 -f https://megengine.org.cn/whl/mge.html
pytest -v test/mge/test_tflite.py
sudo -H python3 -m pip uninstall -y megengine