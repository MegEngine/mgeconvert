# MgeConvert

适用于[`MegEngine`](https://github.com/MegEngine/MegEngine)的各种转换器, 目前支持的框架有：[`Caffe`](https://github.com/BVLC/caffe), [`ONNX`](https://github.com/onnx/onnx), `Cambricon`。

目前适配 ResNet,ResNext,ShuffleNet 如果需要适配其他模型, 可能需要添加更多的opr转换器

MgeConvert前端的部分都在`mge_context`目录下, 可以直接将MegEngine dump出来的计算图转为图结构, 方便后端语言生成

## 当前支持的Op列表

|   |Caffe|ONNX|Cambricon|TFLite|
|-- |-----|----|---------|------|
|abs| ✓ | ✓ | ✓ |  |
|add| ✓ | ✓ | ✓ | ✓ |
|average pool2d| ✓ | ✓ | ✓ |  |
|batchnorm| ✓ | ✓ | ✓ |  |
|broadcast| ✓ | ✓ | ✓ |  |
|ceil| × | ✓ | × |  |
|concat| ✓ | ✓ | ✓ | ✓ |
|conv2d| ✓ | ✓ | ✓ |  |
|convtranspose2d| ✓ | ✓ | ✓ |  |
|div(true_div)| ✓ | ✓ | ✓ | ✓ |
|exp| ✓ | ✓ | ✓ |  |
|elemwise max|  ✓ | ✓ | ✓ | ✓ |
|floor| × | ✓ | ✓ |  |
|log| ✓ | ✓ | ✓ |  |
|matrix mul| ✓ | ✓ | ✓ |  |
|max pool2d| ✓ | ✓ | ✓ |  |
|mul| ✓ | ✓ | ✓ | ✓ |
|pow| ✓ | ✓ | ✓ |  |
|reduce max| ✓ | ✓ | ✓ | ✓ |
|reduce sum| ✓ | ✓ | ✓ | ✓ |
|relu| ✓ | ✓ | ✓ |  |
|reshape| ✓ | ✓ | ✓ | ✓ |
|sigmoid| ✓ | ✓ | ✓ |  |
|softmax| ✓ | ✓ | ✓ |  |
|sub| ✓ | ✓ | ✓ | ✓ |
|slice(subtensor)| ✓ | ✓ | ✓ |  |
|squeeze(axis_add_remove)| ✓ | ✓ | ✓ |  |
|tanh| ✓ | ✓ | ✓ |  |
|typecvt|  ✓ | ✓ | ✓ |  |
|transpose(dimshuffle)| ✓ | ✓ | ✓ |  |


## 安装说明

### 依赖环境安装

* Caffe转换器

  使用Caffe转换器需要编译caffe protobuf, 执行`mgeconvert/caffe_converter/init.sh`下载并编译`caffe.proto`
  ```bash
  ./mgeconvert/caffe_converter/init.sh
  ```
  Caffe 转换器依赖 protobuf（版本>=3.2.0)
  ```bash
  pip3 install protobuf
  ```
* ONNX转换器

  ONNX转换器依赖onnx(1.7.0), protobuf
  ```bash
  pip3 install onnx==1.7.0
  pip3 install protobuf
  ```
  [测试用例](test/test_onnx.py) 中使用了onnxruntime验证导出模型的推理结果是否正确
  ```bash
  pip3 install onnxruntime==1.4.0
  ```

* 寒武纪转换器

  寒武纪转换器目前支持的硬件平台仅包括MLU270智能加速卡系列，使用前请事先安装MLU270智能加速卡和配套SDK并配置好安装路径 `NEUWARE_HOME=/path/to/cambricon`。SDK版本需在1.2.5以上，更多信息请联系寒武纪。

  安装cmake、swig和python-dev

  ```bash
  sudo apt install cmake swig python3-dev
  ```

  以下两个步骤将在安装本仓库时自动执行。

  从numpy官方库下载`numpy.i`，并放入路径`mgeconvert/cambricon_converter/swig`中。

  ```bash
  wget -P mgeconvert/cambricon_converter/swig \
  https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i
  ```

  编译得到python接口。

  ```bash
  cd mgeconvert/cambricon_converter
  mkdir build && cd build
  cmake ..
  make && make develop
  ```

  [测试用例](test/test_cambricon.py) 中可以完成相应的转换测试，[子目录](mgeconvert/cambricon_converter/README.md)获取更多寒武纪转换器相关信息。

* TFLite 转换器

测试过的 op:

* [ ] elemwise
* [ ] reduce
* [ ] reshape
* [ ] concat
* [ ] pool
* [ ] conv
* [ ] relu6
* [ ] activation
* [ ] depthwise-conv
* [ ] fc
* [ ] softmax
* [ ] deconv
* [ ] resize

还没测试的 op:

* [ ] pad
* [ ] typecvt
* [ ] mtk deconv
* [ ] elemwisemultitype

待处理问题：

- flatbuffer 分支是否需要修改

- [done] pyflexbuffers 仓库有没有可能集成进来

### MgeConvert安装

使用setup.py安装MgeConvert。

默认不会安装寒武纪转换器，否则请先设置环境变量 `export USE_CAMBRICON_CONVERTER=ON`。

```bash
pip3 install .
```

## 转换器使用说明

* 使用MegEngine python api (MegEngine jit.trace) `trace()`和`dump()`导出静态图模型，具体使用方法见[MegEngine文档](https://megengine.org.cn/doc/advanced/trace_and_dump.html)。

* 转caffe
 
  执行 [convert_caffe.py](mgeconvert/utils/convert_caffe.py)将MegEngine模型转为Caffe模型，具体使用方法如下：
  ```bash
  python3 -m mgeconvert.utils.convert_caffe -i $MGE_MODEL -c $PROTOTXT -b $CAFFEMODEL [--end_point $ENDPOINT]
  ```
  转换成功后将产生存储模型结构的文本文件`$PROTOTXT`,和存储模型参数的文件 `$CAFFEMODEL`。可以通过设置 --end_point 指定转换到模型某个节点（tensor)为止，多个tensor的名字用`;`分隔，未指定时转换整个模型。

* 转onnx

  执行[convert_onnx.py](mgeconvert/utils/convert_onnx.py)将MegEngine模型转为ONNX模型，具体使用方法如下：
  ```bash
  python3 -m mgeconvert.utils.convert_onnx -i $MGE_MODEL -o $ONNX_MODEL [--opset $OPSET] [--graph $GRAPH] [--end_point $ENDPOINT]
  ```
  转换成功将产生onnx模型文件`$ONNX_MODEL`。可以通过--opset 指定onnx opset版本，当前支持7-12版本的opset，未指定时opset版本为8。可以通过 --graph 设置计算图名，未指定时使用 "graph" 作为计算图名。可以通过设置 --end_point 指定转换到模型某个节点（tensor)为止，多个tensor的名字用`;`分隔，未指定时转换整个模型。

* 转寒武纪

  执行[convert_cambricon.py](mgeconvert/utils/convert_cambricon.py)将MegEngine模型转为寒武纪模型，具体使用方法如下：
  ```bash
  python3 -m mgeconvert.utils.convert_cambricon -i $MGE_MODEL -o $CAMBRICON_MODEL [-b $BATCH_SIZE] [-c $CORE_NUMBER] [-t $DATA_TYPE]
  ```
  转换成功将产生寒武纪离线模型文件`$CAMBRICON_MODEL`以及相应的后缀为`_twins`的描述文件，通过这个描述文件可以得到寒武纪离线模型的输入、输出、布局、批大小等信息。
