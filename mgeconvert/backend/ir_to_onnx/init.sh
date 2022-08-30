#!/bin/bash -e

ADD_USER=""
if [[ $1 == "False" ]]; then
    ADD_USER="--user"
fi

PYTHON3=$2

$PYTHON3 -m pip install "onnx>=1.7.0,<1.12.0" $ADD_USER
$PYTHON3 -m pip install onnxoptimizer==0.2.7 $ADD_USER
$PYTHON3 -m pip install protobuf $ADD_USER
