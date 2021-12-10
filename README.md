# MgeConvert

适用于 [MegEngine](https://github.com/MegEngine/MegEngine) 的各种转换器, 目前支持的框架有 [Caffe](https://github.com/BVLC/caffe)、[ONNX](https://github.com/onnx/onnx) 和 TFLite。

MgeConvert转换工具位于converters目录下，可直接调用其中的脚本将MegEngine导出的mge/TracedModule模型转换为第三方模型文件。

MgeConvert转换器的结构包含前端、中间表示（IR）、后端三个部分：
1. 前端的部分位于 `frontend` 目录下, 支持 mge 和 traced module 模型格式，可以将 MegEngine 序列化出来的计算图转为IR图结构
2. IR部分位于 `converter_ir`目录下，包含图和 IR 算子定义、对计算图做变换的 transform rules 以及对量化模型处理的量化器
3. 后端的部分位于 `backend` 目录下，包含caffe、onnx、tflite的转换器，可以将IR图结构转换为第三方框架的模型文件

目前支持的模型包括 ResNet、ResNext、ShuffleNet 等，如果需要适配其他模型, 可能需要添加更多的算子支持。

## Feature 支持说明

- :white_check_mark: 已支持，并完成测试
- :memo: 未支持，或尚未测试完全
- :boom: 明确不支持

| TracedModule        | tflite             | caffe              | onnx               |
|---------------------|--------------------|--------------------|--------------------|
| QAT                 | :white_check_mark: | :white_check_mark: | :memo:             |
| Quantized           | :white_check_mark: | :boom:             | :memo:             |
| Float32             | :white_check_mark: | :white_check_mark: | :white_check_mark: |

| Mge                 | tflite             | caffe              | onnx               |
|---------------------|--------------------|--------------------|--------------------|
| QAT                 | :boom:             | :boom:             | :boom:             |
| Quantized           | :memo:             | :boom:             | :memo:             |
| Float32             | :white_check_mark: | :white_check_mark: | :white_check_mark: |


## 安装方式

### 依赖说明

MgeConvert 基于 MegEngine 工作，因此确保您的电脑已经安装 MegEngine(>=1.0)。

1. caffe

 - Python packages: protobuf>=3.11.0

2. onnx

 - Python packages: protobuf, onnx>=1.7.0, onnxoptimizer

3. tflite

 - Python packages: pybind11==2.6.2
 - third party: [flatbuffers](https://github.com/google/flatbuffers.git)==1.12.0

> :warning: 安装时以上依赖覆盖本地版本，建议在虚拟环境中安装mgeconvert

如果安装过0.5.0及之前版本的mgeconvert，需要先卸载旧版本：

```bash
sudo pip3 uninstall mgeconvert
```


### pip 安装

- 以 caffe 为例，下面这条指令将通过``pip``包管理器安装开发版本的 caffe 转换器并处理相关依赖：

```bash
python3 -m pip install git+https://github.com/MegEngine/mgeconvert.git --user --install-option="--targets=caffe"
```

> 建议指定版本号安装release版本的转换器，如安装0.4.2版本：

```bash
python3 -m pip install git+https://github.com/MegEngine/mgeconvert.git@v0.4.2 --user --install-option="--targets=caffe"
```

> :warning: 如果需要转换``TracedModule``模型，请安装0.5.0以上版本


- ``--targets`` 的可选值有 ``caffe``、``onnx``、``tflite`` 和 `all`。

`all` 代表安装全部转换器。可选值支持组合传入，比如 ``--targets=caffe,tflite`` 。

- ``tflite`` 转换器的schema默认使用r2.3版本，支持使用参数 ``tfversion`` 选择tflite schema的版本, 例如：

```bash
--install-option="--targets=tflite --tfversion=r2.4"
```

### 源代码安装

安装选项说明同上，以 caffe 为例，下面的命令将安装0.4.2版本的caffe转换器：

```bash
git clone https://github.com/MegEngine/mgeconvert.git@v0.4.2
cd mgeconvert
pip3 install . --user --install-option="--targets=caffe"
```

## 使用方式

转换器按输入模型格式主要分为两种：
1. 使用megengine jit.trace [dump](https://megengine.org.cn/doc/stable/zh/user-guide/model-development/jit/dump.html#dump) 出来的序列化模型，这类模型的转换器以 `mge_to` 命名
2. [TracedModule](https://megengine.org.cn/doc/stable/zh/user-guide/deployment/traced_module/index.html#) 导出的序列化模型，这类模型的转换器以 `tracedmodule_to` 命名

### 1. 命令行使用

执行脚本位于 ``~/.local/bin`` 文件夹内，使用前需要将此路径加入到环境变量 ``PATH`` 中。

命令行支持命令补全，执行 `convert --init` 即可使用。

查询支持的转换框架，结果取决于安装时的 ``--install-option``：

```bash
convert -h
```

以 mge模型转 caffe 为例，查询转换参数：

```bash
convert mge_to_caffe -h
```

#### caffe模型转换

##### float模型转换

- 转换mge模型的参考命令：

```bash
convert mge_to_caffe -i model.mge -c out.prototxt -b out.caffemodel
```

- 转换 TracedModule 模型的参考命令：

```bash
convert tracedmodule_to_caffe -i model.tm -c out.prototxt -b out.caffemodel
```

##### QAT模型转换
mgeconvert 支持将 QAT TracedModule 模型转换到caffe：
- QAT模型转caffe默认会导出量化参数文件，通过 `quantize_file_path` 指定量化参数文件路径：

```bash
convert tracedmodule_to_caffe -i qat_model.tm -c out.prototxt -b out.caffemodel --quantize_file_path quant_params.json
```

- 添加 `param_fake_quant` 参数可选择对模型参数进行假量化：

```bash
convert tracedmodule_to_caffe -i qat_model.tm -c out.prototxt -b out.caffemodel --quantize_file_path quant_params.json --param_fake_quant
```

- 如果QAT模型中没有QuantStub对输入数据进行量化处理，可以在转换时指定输入数据的量化类型、scale和zero_point量化参数 ：

```bash
convert tracedmodule_to_caffe -i qat_model.tm -c out.prototxt -b out.caffemodel --quantize_file_path quant_params.json --input_data_type quint8 --input_scales 0.125 --input_zero_points 128
```


#### tflite模型转换

TFlite转换器支持 float32 和量化的 TracedModule 转换。

##### float模型转换

转换float模型的命令参考：

```bash
convert mge_to_tflite -i model.mge -o out.tflite
```

```bash
convert tracedmodule_to_tflite -i tracedmodule.tm -o out.tflite
```

##### QAT模型转换

- 对于QAT模型，可以通过添加tracedmodule_to_tflite转换器中的 `require_quantize` 选项，转换出tflite支持的量化数据类型（int8/uint8/int16/int32）量化后的Quantized 模型:

```bash
convert tracedmodule_to_tflite -i tracedmodule.tm -o out.tflite --require_quantize
```

也可不设置 `require_quantize` 选项，转换出float32模型和量化参数文件。

```bash
convert tracedmodule_to_tflite -i tracedmodule.tm -o out.tflite --quantize_file_path quant_params.json
```

- 对于QAT模型，还可以通过设置 `param_fake_quant` 参数来选择是否对参数进行假量化。

```bash
convert tracedmodule_to_tflite -i tracedmodule.tm -o out.tflite --quantize_file_path quant_params.json --param_fake_quant
```

- 如果QAT模型中没有QuantStub对输入数据进行量化处理，可以在转换时指定输入数据的量化类型、scale和zero_point量化参数,如果有多个scale、zero point用逗号隔开：

```bash
convert tracedmodule_to_tflite -i tracedmodule.tm -o out.tflite --input_data_type quint8 --input_scales 0.125,0.125 --input_zero_points 128,128 --require_quantize
```

#### onnx模型转换

mgeconvert 转 onnx 模型支持 [opset](https://github.com/onnx/onnx/blob/master/docs/Operators.md) 7~12 的转换。

目前只支持float模型转到onnx，转换命令参考：

```bash
convert mge_to_onnx -i model.mge -o out.onnx
```

```bash
convert tracedmodule_to_onnx -i tracedmodule.tm -o out.onnx
```

### 2. python接口使用

可参考[wiki](https://github.com/MegEngine/mgeconvert/wiki/Mgeconvert-Python-Api-Doc)中的例子。

## 常见问题说明

1. 安装时出现类似报错：

```
error removing /home/user/.local/lib/python3.6/site-packages/mgeconvert-0.5.0-py3.6.egg-info: 
[Errno 13] Permission denied: '/home/user/.local/lib/python3.6/site-packages/mgeconvert-0.5.0-py3.6.egg-info/PKG-INFO'
```

这是使用sudo安装过旧版本出现的权限问题，先卸载旧版本再安装：

```bash
sudo pip3 uninstall mgeconvert
```

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
| resize                   | ✓<br/>✓ | ×<br/>× | ✓<br/>✓ |
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
