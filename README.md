# MgeConvert

适用于 [MegEngine](https://github.com/MegEngine/MegEngine) 的各种转换器, 目前支持的框架有 [Caffe](https://github.com/BVLC/caffe)、[ONNX](https://github.com/onnx/onnx) 和 TFLite。

MgeConvert转换工具位于converters目录下，可直接调用其中的脚本将MegEngine导出的mge/TracedModule模型转换为第三方模型文件。

MgeConvert转换器的结构包含前端、中间表示（IR）、后端三个部分：
1. 前端的部分位于 `frontend` 目录下, 支持 mge 和 traced module 模型格式，可以将 MegEngine 序列化出来的计算图转为IR图结构
2. IR部分位于 `converter_ir`目录下，包含图和 IR 算子定义、对计算图做变换的 transform rules 以及对量化模型处理的量化器
3. 后端的部分位于 `backend` 目录下，包含caffe、onnx、tflite的转换器，可以将IR图结构转换为第三方框架的模型文件

目前支持的模型包括 ResNet、ResNext、ShuffleNet 等，如果需要适配其他模型, 可能需要添加更多的算子支持。

## 安装方式

MgeConvert 基于 MegEngine 工作，因此确保您的电脑已经安装 MegEngine(>=1.0)。

### pip 包管理器安装

以 caffe 为例，下面这条指令将通过``pip``包管理器安装开发版本的 caffe 转换器并处理相关依赖：

```bash
python3 -m pip install git+https://github.com/MegEngine/mgeconvert.git --user --install-option="--targets=caffe"
```

``--targets`` 的可选值有 ``caffe``、``onnx``、``tflite`` 和 ``all``。

``tflite`` 转换器的schema默认使用r2.3版本，支持使用参数 ``tfversion`` 选择tflite schema的版本, 例如：

```bash
--install-option="--targets=tflite --tfversion=r2.4"
```

``all`` 代表安装全部转换器，不建议使用。可选值支持组合传入，比如 ``--targets=caffe,onnx`` 。

建议使用时指定版本号安装release版本的转换器，如安装0.4.2版本：

```bash
python3 -m pip install git+https://github.com/MegEngine/mgeconvert.git@v0.4.2 --user --install-option="--targets=caffe"
```
> :warning: 如果需要转换``TracedModule``模型，请安装v0.5.0以上版本

### 源代码安装

安装选项说明同上，以 caffe 为例，下面的命令将安装0.4.2版本的caffe转换器：

```bash
git clone https://github.com/MegEngine/mgeconvert.git@v0.4.2
python3 setup.py install --user --install-option="--targets=caffe"
```

## 使用方式

执行脚本位于 ``~/.local/bin`` 文件夹内，使用前需要将此路径加入到环境变量 ``PATH`` 中。

查询支持的转换框架，结果取决于安装时的 ``--install-option``：

```bash
convert -h
```

以 mge模型转 caffe 为例，查询转换参数：

```bash
convert mge_to_caffe -h
```

### Feature 支持说明

- :white_check_mark: 已支持，并完成测试
- :memo: 未支持，或尚未测试完全
- :boom: 明确不支持

| TracedModule        | tflite             | caffe              | onnx               |
|---------------------|--------------------|--------------------|--------------------|
| QAT                 | :white_check_mark: | :memo:             | :memo:             |
| Quantized           | :white_check_mark: | :boom:             | :memo:             |
| Float32             | :white_check_mark: | :white_check_mark: | :white_check_mark: |

| Mgo                 | tflite             | caffe              | onnx               |
|---------------------|--------------------|--------------------|--------------------|
| QAT                 | :boom:             | :boom:             | :boom:             |
| Quantized           | :memo:             | :boom:             | :memo:             |
| Float32             | :white_check_mark: | :white_check_mark: | :white_check_mark: |

### TFLite转换
TFlite转换器支持 Float32 和量化的 TracedModule 转换。
对于QAT模型，可以通过设置tracedmodule_to_tflite转换器中的 `require_quantize=True` 选择转换出tflite支持的量化数据类型（int8/uint8/int16/int32）量化后的Quantized 模型:

```bash
convert tracedmodule_to_tflite -i tracedmodule.tm -o out.tflite --require_quantize
```

也可设置 `require_quantize=False` 选择转换出float32模型和量化参数文件。

```bash
convert tracedmodule_to_tflite -i tracedmodule.tm -o out.tflite --quantize_file_path "quant_params.json"
```

对于后者，还可以通过设置 `param_fake_quant` 参数来选择是否对参数进行假量化。

如果模型中没有QuantStub对输入数据进行量化处理，可以在转换时指定输入数据的量化类型、scale和zero_point量化参数 ：

```bash
convert tracedmodule_to_tflite -i tracedmodule.tm -o out.tflite --input_data_type "quint8" --input_scales 0.125 --input_zero_points 128 --require_quantize
```

## 依赖说明

1. caffe

 - Python packages: protobuf>=3.2.0

2. onnx

 - Python packages: protobuf, onnx==1.7.0

3. tflite

 - Python packages: pybind11==2.6.2, tensorflow==2.4.0
 - third party: [flatbuffers](https://github.com/google/flatbuffers.git)


## 算子支持列表

| tracemodule:rocket:<br/>mgo:fire:      | TFLite  | Caffe   | ONNX    |
|--------------------------|---------|---------|---------|
| abs                      | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| average pool2d           | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| batchnorm                | ×<br/>× | ✓<br/>✓ | ✓<br/>✓ |
| broadcast                | ×<br/>× | ✓<br/>✓ | ✓<br/>✓ |
| ceil                     | ×<br/>× | ×<br/>× | ✓<br/>✓ |
| concat                   | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| conv2d                   | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| convtranspose2d          | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| div(true_div)            | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| exp                      | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| elemwise max             | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| floor                    | ×<br/>× | ×<br/>× | ✓<br/>✓ |
| log                      | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| matrix mul               | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| max pool2d               | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| mul                      | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| pow                      | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| reduce max               | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| reduce min               | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| reduce mean              | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| reduce sum               | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| relu                     | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| relu6                    | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| reshape                  | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| resize                   | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| sigmoid(logistic)        | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| softmax                  | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| leaky_relu               | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| sub                      | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| slice(subtensor)         | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| squeeze(axis_add_remove) | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| tanh                     | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| typecvt                  | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| transpose(dimshuffle)    | ✓<br/>✓ | ✓<br/>✓ | ✓<br/>✓ |
| AdaptiveAvgPool2d        | ×<br/>× | ✓<br/>✓ | ✓<br/>✓ |
| flatten                  | ×<br/>× | ×<br/>× | ✓<br/>✓ |
