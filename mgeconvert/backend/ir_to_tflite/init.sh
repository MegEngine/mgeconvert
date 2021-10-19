#!/bin/bash -e
basepath=$(cd `dirname $0`; pwd)

    
which flatc  && FLATC_VERSION="$(flatc --version)" || FLATC_VERSION=""
echo ${FLATC_VERSION}
if python3 -c "import flatbuffers">/dev/null 2>&1; then
    FLAT_BUFFER_VERSION="$(python3 -m pip show flatbuffers | grep Version)"
else
    FLAT_BUFFER_VERSION=""
fi
echo ${FLAT_BUFFER_VERSION}


if [[ "$FLATC_VERSION" != "flatc version 1.12.0" || "$FLAT_BUFFER_VERSION" != "Version: 1.12" ]]; then
    sudo rm -rf /tmp/flatbuffers
    git clone https://github.com/google/flatbuffers.git -b v1.12.0 /tmp/flatbuffers
fi

if [[ "$FLATC_VERSION" != "flatc version 1.12.0" ]]; then
    sudo python3 -m pip uninstall flatbuffers -y
    sudo rm -rf /usr/local/bin/flatc
    sudo rm -rf /usr/local/lib/libflatbuffers*
    # build flatbuffers
    echo "building flatbuffers..."
    cd /tmp/flatbuffers
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_SHAREDLIB=on -DCMAKE_INSTALL_PREFIX=/usr/local
    make; sudo make install
fi


export PATH=$PATH:/usr/local/bin
# build tflite interface from schema.fbs
echo "building tflite schema..."
cd /tmp
sudo rm -f schema.fbs
tf_version=$1
if [ ! -n "$1" ] ;then
    tf_version="r2.3"
fi
echo "Use TFLite $tf_version!"
wget https://raw.githubusercontent.com/tensorflow/tensorflow/$tf_version/tensorflow/lite/schema/schema.fbs
sudo flatc --python schema.fbs
sudo chmod 755 /tmp/tflite
cp -r /tmp/tflite $basepath

# build pyflatbuffers
if [[ "$FLAT_BUFFER_VERSION" != "Version: 1.12" ]]; then
    sudo python3 -m pip uninstall flatbuffers -y
    echo "building pyflexbuffers..."
    export VERSION=1.12
    cd /tmp/flatbuffers/python
    sudo python3 setup.py install
fi


sudo python3 -m pip install pybind11==2.6.2

# try to find libflatbuffers.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# using pyflexbuffers
cd $basepath/pyflexbuffers
PYBIND11_HEADER=$(python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYTHON_STDLIB=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])")

g++ fbconverter.cc --std=c++14 -fPIC --shared -I${PYBIND11_HEADER} -I${PYTHON_INCLUDE} -L${PYTHON_STDLIB}  -lflatbuffers -o fbconverter.so
