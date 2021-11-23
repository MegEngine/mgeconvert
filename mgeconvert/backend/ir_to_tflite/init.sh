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

if [ ! -d /tmp/mgeconvert ]; then
    mkdir /tmp/mgeconvert
fi

TMP_DIR="/tmp/mgeconvert/flatbuffers"

if [[ "$FLATC_VERSION" != "flatc version 1.12.0" || "$FLAT_BUFFER_VERSION" != "Version: 1.12" ]]; then
    rm -rf $TMP_DIR
    git clone https://github.com/google/flatbuffers.git -b v1.12.0 $TMP_DIR
fi

if [[ "$FLATC_VERSION" != "flatc version 1.12.0" ]]; then
    python3 -m pip uninstall flatbuffers -y
    rm -rf $HOME/.local/bin/flatc
    rm -rf $HOME/.local/lib/libflatbuffers*
    # build flatbuffers
    echo "building flatbuffers..."
    cd $TMP_DIR
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_SHAREDLIB=on -DCMAKE_INSTALL_PREFIX=$HOME/.local
    make -j; make install
fi


export PATH=$PATH:$HOME/.local/bin
# build tflite interface from schema.fbs
echo "building tflite schema..."
cd /tmp/mgeconvert
rm -f schema.fbs
tf_version=$1
if [ ! -n "$1" ] ;then
    tf_version="r2.3"
fi
echo "Use TFLite $tf_version!"
wget https://raw.githubusercontent.com/tensorflow/tensorflow/$tf_version/tensorflow/lite/schema/schema.fbs
flatc --python schema.fbs
chmod 777 /tmp/mgeconvert/tflite
cp -r /tmp/mgeconvert/tflite $basepath

# build pyflatbuffers
if [[ "$FLAT_BUFFER_VERSION" != "Version: 1.12" ]]; then
    python3 -m pip uninstall flatbuffers -y
    echo "building pyflexbuffers..."
    export VERSION=1.12
    cd $TMP_DIR/python
    python3 setup.py install --user
fi


python3 -m pip install pybind11==2.6.2 --user

# using pyflexbuffers
cd $basepath/pyflexbuffers
PYBIND11_HEADER=$(python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYTHON_STDLIB=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])")

g++ fbconverter.cc --std=c++14 -fPIC --shared -I$HOME/.local/include -I${PYBIND11_HEADER} -I${PYTHON_INCLUDE} -L${PYTHON_STDLIB} -L$HOME/.local/lib  -lflatbuffers -o fbconverter.so

rm -r /tmp/mgeconvert