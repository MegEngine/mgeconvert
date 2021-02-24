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
# FIXME: fix this repo later
cd $basepath
rm -rf /tmp/pyflexbuffers
git clone git@git-core.megvii-inc.com:brain-sdk/pyflexbuffers.git /tmp/pyflexbuffers
cd /tmp/pyflexbuffers
sudo python3 setup.py install
