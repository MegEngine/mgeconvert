#!/bin/bash -e
hash wget || (echo "please install wget package" && exit -1)

cd $(dirname $0)

wget https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i -P swig/

mkdir build && cd build
cmake .. && make
mv _cambriconLib.so ../lib/cnlib/
mv cambriconLib.py ../lib/cnlib/
