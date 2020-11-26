#!/bin/bash -e

set -e

pushd ci >/dev/null
pip install -q -r requires-style.txt
./format.sh -d
popd >/dev/null
