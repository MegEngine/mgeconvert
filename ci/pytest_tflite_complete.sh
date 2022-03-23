#!/bin/bash -e

set -e

./mgeconvert/backend/ir_to_tflite/init.sh

# try to find libflatbuffers.so
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH:
sudo python3 -m pip uninstall flatbuffers -y
sudo python3 -m pip install tensorflow==2.5.0

function version_gt() { test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" != "$1"; }

MGE_VERSIONS="1.9.0 1.8.1 1.7.0 1.6.0 1.5.0 1.4.0 1.3.0 1.2.0 1.1.0 1.0.0"
for VER in $MGE_VERSIONS
do
    sudo -H python3 -m pip install -q megengine==$VER -f https://megengine.org.cn/whl/mge.html
    pytest test/mge/test_tflite.py
    if version_gt $VER "1.5.0";then
        pytest test/traced_module/test_tflite.py
        pytest test/traced_module/test_qat_tflite.py
    fi
    sudo -H python3 -m pip uninstall -y megengine
done
