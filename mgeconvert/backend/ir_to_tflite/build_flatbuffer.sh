#!/bin/bash -e
basepath=$(cd `dirname $0`; pwd)


if [ ! -d /tmp/mgeconvert ]; then
    mkdir /tmp/mgeconvert
fi
TMP_DIR="/tmp/mgeconvert/flatbuffers"
rm -rf $TMP_DIR

# build flatbuffers
git clone https://github.com/google/flatbuffers.git -b v1.12.0 $TMP_DIR
echo "building flatbuffers..."
cd $TMP_DIR
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_SHAREDLIB=on -DCMAKE_INSTALL_PREFIX=${basepath}/pyflexbuffers
make -j; make install
echo "build flatbuffers done"

rm -rf $TMP_DIR
