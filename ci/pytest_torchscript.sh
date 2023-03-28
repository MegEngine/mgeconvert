#!/bin/bash -e

set -e

python3 -m pip install -q -r ci/requires-test.txt

# hack for megengine.hub
# https://github.com/MegEngine/Models/blame/master/requirements.txt
pip3 install ftfy imageio youtokentome regex==2020.10.15

pip3 install torch>=1.10

MGE_VERSIONS="1.12.2 1.9.0 1.8.1 1.7.0 1.6.0"
for VER in $MGE_VERSIONS
do
    sudo -H python3 -m pip install -q megengine==$VER -f https://megengine.org.cn/whl/mge.html
    pytest test/traced_module/test_torchscript.py
    sudo -H python3 -m pip uninstall -y megengine
done
