from test.utils import ConvOpr, dump_mge_model

import megengine as mge
import numpy as np
from megengine.core.tensor import dtype
from megengine.quantization.quantize import quantize_qat
from megengine.traced_module import trace_module

if __name__ == "__main__":
    net = ConvOpr("normal")
    traced_module = trace_module(net, mge.tensor(net.data))
    mge.save(traced_module, "float_model.tm")
    dump_mge_model(net, net.data, "float_model")

    qat_net = quantize_qat(net)
    inp_dtype = dtype.qint8(16.0 / 128)
    data = mge.tensor(np.random.random((1, 3, 224, 224))) * 16
    data = data.astype(inp_dtype)
    inp = mge.tensor(dtype.convert_from_qint8(data.numpy()))
    inp.qparams.scale = mge.tensor(dtype.get_scale(inp_dtype))
    inp.qparams.dtype_meta = dtype._builtin_quant_dtypes["qint8"]

    qat_module = trace_module(qat_net, inp)
    mge.save(qat_module, "qat_model.tm")
