# MgeConvert

适用于 [MegEngine](https://github.com/MegEngine/MegEngine) 的各种转换器, 目前支持的框架有 [Caffe](https://github.com/BVLC/caffe)、[ONNX](https://github.com/onnx/onnx)、Cambricon 和 TFLite。

支持的模型包括 ResNet、ResNext、ShuffleNet 等，如果需要适配其他模型, 可能需要添加更多的算子支持。

MgeConvert 前端的部分位于 `mge_context` 目录下, 可以直接将 MegEngine dump 出来的计算图转为图结构, 方便后端语言生成。


## 安装方式

MgeConvert 基于 MegEngine 工作，因此确保您的电脑已经安装 MegEngine。

以 caffe 为例，下面这条指令将安装 caffe 转换器并处理相关依赖。

```bash
python3 -m pip install git+https://github.com/MegEngine/mgeconvert.git --user --install-option="--targets=caffe"
```

``--targets`` 的可选值有 ``caffe``、``onnx``、``cambricon``、``tflite`` 和 ``all``。

``all`` 代表安装全部转换器，不建议使用。可选值支持组合传入，比如 ``--targets=caffe,onnx`` 。

> :warning: 由于 Cambricon SDK 未对外发布，寒武纪转换器将不会自动安装依赖，请事先查看依赖说明并配置环境。

## 使用方式

执行脚本位于 ``~/.local/bin`` 文件夹内，使用前需要将此路径加入到环境变量 ``PATH`` 中。

查询支持的转换框架，结果取决于安装时的 ``--install-option``：

```bash
convert -h
```

以 caffe 为例，查询转换参数：

```bash
convert caffe -h
```

## 依赖说明

1. caffe

 - Python packages: protobuf>=3.2.0

2. onnx

 - Python packages: protobuf, onnx==1.7.0

3. cambricon

 - Cambricon SDK: CNRT, CNML

4. tflite

 - @yzchen


## 算子支持列表

|   |Caffe|ONNX|Cambricon|TFLite|
|-- |-----|----|---------|------|
|abs| ✓ | ✓ | ✓ | × |
|add| ✓ | ✓ | ✓ | ✓ |
|average pool2d| ✓ | ✓ | ✓ | ✓ |
|batchnorm| ✓ | ✓ | ✓ | × |
|broadcast| ✓ | ✓ | ✓ | × |
|ceil| × | ✓ | × | × |
|concat| ✓ | ✓ | ✓ | ✓ |
|conv2d| ✓ | ✓ | ✓ | ✓ |
|convtranspose2d| ✓ | ✓ | ✓ | ✓ |
|div(true_div)| ✓ | ✓ | ✓ | ✓ |
|exp| ✓ | ✓ | ✓ | ✓ |
|elemwise max|  ✓ | ✓ | ✓ | ✓ |
|floor| × | ✓ | ✓ | × |
|log| ✓ | ✓ | ✓ | × |
|matrix mul| ✓ | ✓ | ✓ | ✓ |
|max pool2d| ✓ | ✓ | ✓ | ✓ |
|mul| ✓ | ✓ | ✓ | ✓ |
|pow| ✓ | ✓ | ✓ | × |
|reduce max| ✓ | ✓ | ✓ | ✓ |
|reduce sum| ✓ | ✓ | ✓ | ✓ |
|relu| ✓ | ✓ | ✓ | ✓ |
|reshape| ✓ | ✓ | ✓ | ✓ |
|sigmoid| ✓ | ✓ | ✓ | × |
|softmax| ✓ | ✓ | ✓ | ✓ |
|leaky_relu| ✓ | × | × | ✓ |
|sub| ✓ | ✓ | ✓ | ✓ |
|slice(subtensor)| ✓ | ✓ | ✓ | × |
|squeeze(axis_add_remove)| ✓ | ✓ | ✓ | × |
|tanh| ✓ | ✓ | ✓ | ✓ |
|typecvt|  ✓ | ✓ | ✓ | ✓ |
|transpose(dimshuffle)| ✓ | ✓ | ✓ | × |
