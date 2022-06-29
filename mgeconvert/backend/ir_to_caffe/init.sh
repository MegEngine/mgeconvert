#!/bin/bash -e

ADD_USER=""
if [[ $1 == "False" ]]; then
    ADD_USER="--user"
fi

PYTHON3=$2

$PYTHON3 -m pip install --no-binary=protobuf "protobuf>=3.11.1" $ADD_USER

hash wget || (echo "please install wget package" && exit -1)
hash protoc || (echo "please install protobuf-compiler package" && exit -1)

cd $(dirname $0)

BUILD_DIR="./caffe_pb"
mkdir -p $BUILD_DIR

echo "Retrieving lastest caffe.proto from github.com"
wget https://github.com/BVLC/caffe/raw/master/src/caffe/proto/caffe.proto -O $BUILD_DIR/caffe.proto

echo "Compiling caffe.proto"
protoc $BUILD_DIR/caffe.proto --python_out=./

touch $BUILD_DIR/__init__.py

echo "Init done"
