#!/bin/bash -e
hash wget || (echo "please install wget package" && exit -1)
hash protoc || (echo "please install protobuf-compiler package" && exit -1)

cd $(dirname $0)

BUILD_DIR="./caffe_pb"
mkdir -p $BUILD_DIR

echo "Retrieving lastest caffe.proto from github.com"
wget https://github.com/BVLC/caffe/raw/master/src/caffe/proto/caffe.proto -O $BUILD_DIR/caffe.proto

echo "Compiling caffe.proto"
protoc $BUILD_DIR/caffe.proto --python_out=./

touch $BUILD_DIR/__init__.pysu

echo "Init done"
