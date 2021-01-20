#!/bin/bash -e

set -e

sudo -H python3 -m pip install -q megengine==1.2.0 -f https://megengine.org.cn/whl/mge.html
pytest test/test_cambricon.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.1.0 -f https://megengine.org.cn/whl/mge.html
pytest test/test_cambricon.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==1.0.0 -f https://megengine.org.cn/whl/mge.html
pytest test/test_cambricon.py
sudo -H python3 -m pip uninstall -y megengine

sudo -H python3 -m pip install -q megengine==0.6.0 -f https://megengine.org.cn/whl/mge.html
pytest test/test_cambricon.py
sudo -H python3 -m pip uninstall -y megengine