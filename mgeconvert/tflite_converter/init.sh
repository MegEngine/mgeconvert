#!/bin/bash -e
basepath=$(cd `dirname $0`; pwd)

# build flatbuffers
sudo rm -rf /tmp/flatbuffers
git clone https://github.com/google/flatbuffers.git /tmp/flatbuffers
cd /tmp/flatbuffers
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_SHAREDLIB=on -DCMAKE_INSTALL_PREFIX=/usr/local
make -j; sudo make install

export PATH=$PATH:/usr/local/bin
# build tflite interface from schema.fbs
cd /tmp
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs
flatc --python schema.fbs
cp -r /tmp/tflite $basepath
echo "======================"
echo $(cat /tmp/tflite/__init__.py)
echo "======================"

# build pyflatbuffers
cd /tmp/flatbuffers/python
sudo python3 setup.py install

sudo python3 -m pip install pybind11==2.6.2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
# using pyflexbuffers
cd $basepath/pyflexbuffers
PYBIND11_HEADER=$(python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYTHON_STDLIB=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])")

g++ fbconverter.cc --std=c++14 -fPIC --shared -I${PYBIND11_HEADER} -I${PYTHON_INCLUDE} -L${PYTHON_STDLIB}  -lflatbuffers -o fbconverter.so
