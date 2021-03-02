#!/bin/bash -e
basepath=$(cd `dirname $0`; pwd)

rm -rf /tmp/flatbuffers
# using self-modified flatbuffer
git clone https://github.com/lcxywfe/flatbuffers.git /tmp/flatbuffers
cd /tmp/flatbuffers
git checkout add-finish-with-file-identifier
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_SHAREDLIB=on
sudo make install

# build tflite interface from schema.fbs
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs
./flatc --python schema.fbs
cd python
sudo python3 setup.py install
cp -r /tmp/flatbuffers/tflite $basepath

# using pyflexbuffers
cd $basepath/pyflexbuffers
PYBIND11_HEADER=$(python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INCLUDE=$(python3 -c "import sysconfig; sysconfig.get_paths()['include']")
PYTHON_STDLIB=$(python3 -c "import sysconfig; sysconfig.get_paths()['stdlib']")

g++ fbconverter.cc --std=c++14 -fPIC --shared -I$PYBIND11_HEADER -I$PYTHON_INCLUDE -L$PYTHON_STDLIB  -lflatbuffers -o fbconverter.so
