#!/bin/bash -e

set -e

wget -P mgeconvert/cambricon_converter/swig https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i

mkdir -p mgeconvert/cambricon_converter/build
cd mgeconvert/cambricon_converter/build
cmake ..
make -j4
make develop
cd ../../..
