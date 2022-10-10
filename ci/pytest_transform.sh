#!/bin/bash -e

set -e

python3 -m pip install -q -r ci/requires-test.txt

sudo -H python3 -m pip install -q megengine==1.9.0 -f https://megengine.org.cn/whl/mge.html
pytest test/traced_module/test_transform.py
