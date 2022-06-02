#!/bin/bash -e
basepath=$(cd `dirname $0`; pwd)


if python3 -c "import flatbuffers">/dev/null 2>&1; then
    FLAT_BUFFER_VERSION="$(python3 -m pip show flatbuffers | grep Version)"
else
    FLAT_BUFFER_VERSION=""
fi
echo ${FLAT_BUFFER_VERSION}

# install flatbuffers
if [[ "$FLAT_BUFFER_VERSION" != "Version: 1.12" ]]; then
    python3 -m pip uninstall flatbuffers -y
    echo "install flatbuffers..."
    python3 -m pip install flatbuffers==1.12 --user
fi

if [ ! -d /tmp/mgeconvert ]; then
    mkdir /tmp/mgeconvert
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
$basepath/pyflexbuffers/bin/flatc --python schema.fbs
chmod 777 /tmp/mgeconvert/tflite
cp -r /tmp/mgeconvert/tflite $basepath


python3 -m pip install pybind11==2.6.2 --user

# using pyflexbuffers
cd $basepath/pyflexbuffers
PYBIND11_HEADER=$(python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYTHON_STDLIB=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])")

g++ fbconverter.cc --std=c++14 -fPIC --shared -I$basepath/pyflexbuffers/include -I${PYBIND11_HEADER} -I${PYTHON_INCLUDE} -L${PYTHON_STDLIB} -L$basepath/pyflexbuffers/lib  -lflatbuffers -o fbconverter.so

rm -rf /tmp/mgeconvert