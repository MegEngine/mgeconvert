# 寒武纪转换器

## 模型要求

寒武纪模型在MLU270上推理时要求含乘法的算子采取量化(`QInt8`)计算，包括`Conv`、`MatMul`、`Lrn`等，其他算子采用浮点数(`Float32`或`Float16`)计算。因此转换前的MegEngine模型也要具备这个特征才能保证转换后模型的精度不下降。

对应算子的输入和输出类型要求：

- 量化算子：src(qint8) + filter(qint8) + bias(qint32) -> dst(qint32)
- 其他算子：src(fp32) -> dst(fp32)

参考[测试用例](test/quantization_utils.py)中的`QuantizationLinearOpr`和`QuantizationConvBnOpr`来完成此类模型的搭建。


## 添加未支持算子

模型转换分成两个阶段，增加算子也遵循这个规则。

- 解析MegEngine模型，可参考其他算子的实现
    1. 在上级目录 `mge_context/mge_op.py` 中增加相应的算子
- 搭建寒武纪模型
    1. 在本目录 `swig/cambricon.i` 中封装寒武纪SDK提供的算子接口
    2. 在本目录 `lib/operators.py` 中实现该算子类
    3. 在本目录 `converter.py` 中增加相应的转换函数
    4. 在根目录 `test/utils` 中添加测试

## 对分经验

如果转换成功但对分失败，可在文件`converter.py`的`convert`函数中设置`end_op`进行二分法对分，快速定位是错误算子。