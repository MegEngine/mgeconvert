import random
from mgeconvert.converter_ir.ir_op import BatchNormalizationOpr
from mgeconvert.backend.ir_to_trt.utils import mge_dtype_from_trt
from test.utils import (
    AdaptiveAvgPool2dOpr,
    ActiveOpr,
    FlattenOpr,
    BnOpr,
    ConcatOpr,
    ConvOpr,
    ElemwiseOpr,
    LinearOpr,
    PoolOpr,
    ReduceOpr,
    ReshapeOpr,
    SoftmaxOpr,
    SqueezeOpr,
    SubtensorOpr,
    TransposeOpr,
    TypeCvtOpr,
    XORNet,
    dump_mge_model,
)
import megengine as mge
import megengine.hub
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from mgeconvert.converters.mge_to_trt import mge_to_trt
from megengine.module.external import TensorrtRuntimeSubgraph
import pytest
from basecls.utils import registers, set_nccl_env, set_num_threads
from basecls.utils import recursive_update, registers

max_error = 1e-4
tmp_file = "test_model"

# def _test_convert_result(
#     inputs, fpath, mge_result, max_err,
# ):

#     mge_to_trt(
#         fpath + ".mge",
#         output=tmp_file + ".trt",
#     )

#     with open(tmp_file + ".trt", "rb") as f:
#         engine = f.read()
#         trt_module = TensorrtRuntimeSubgraph(engine)
#         pred_trt = trt_module(mge.tensor(inputs))

#     assert pred_trt[0].shape == mge_result.shape
#     assert pred_trt[0].dtype == mge_result.dtype

#     np.testing.assert_allclose(pred_trt[0], mge_result, atol=max_err)
#     print("success!")


def _test_convert_result(
    inputs, fpath, mge_result, max_err,
):

    mge_to_trt(
        fpath + ".mge",
        output=tmp_file + ".trt",
    )

    with open(tmp_file + ".trt", "rb") as f:
        engine = f.read()
    
    context, h_input, h_output, stream, d_input, d_output = init(engine, inputs.shape[0])
    np.copyto(h_input, inputs)
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(
        stream_handle=stream.handle, 
        bindings=[int(d_input), int(d_output)],
         batch_size=inputs.shape[0])
    
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()
    print(h_output)
    print(mge_result)
    assert h_output.shape == mge_result.shape
    assert h_output.dtype == mge_result.dtype

    np.testing.assert_allclose(h_output, mge_result, atol=max_err)
    print("success!")


def init(engine, batch_size):
    with trt.Logger(trt.Logger.ERROR) as logger, trt.Runtime(logger) as runtime:

        engine = runtime.deserialize_cuda_engine(engine)
        # 1. Allocate some host and device buffers for inputs and outputs:
        h_input_shape = (batch_size,) + tuple(engine.get_binding_shape(0))
        # h_input = np.empty(engine.get_binding_shape(0), dtype=trt.nptype(trt.float32))
        h_input = np.empty(h_input_shape, dtype=trt.nptype(trt.float32))

        h_output_shape = (batch_size,) + tuple(engine.get_binding_shape(1))
        # h_output = np.empty(engine.get_binding_shape(1), dtype=trt.nptype(trt.float32))
        h_output = np.empty(h_output_shape, dtype=trt.nptype(trt.float32))
        # Allocate device memory for inputs and outputs.
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()
        context = engine.create_execution_context()
        return context, h_input, h_output, stream, d_input, d_output


# def _test_convert_result(
#     inputs, fpath, mge_result, max_err,
# ):

#     mge_to_trt(
#         fpath + ".mge",
#         output=tmp_file + ".trt",
#     )

#     with open(tmp_file + ".trt", "rb") as f:
#         engine = f.read()
    
#     context, h_input, h_output, stream, d_input, d_output = init_buffer(engine)

#     np.copyto(h_input, inputs)
#     cuda.memcpy_htod_async(d_input, h_input, stream)
#     # Run inference.
#     context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
#     # Transfer predictions back from the GPU.
#     cuda.memcpy_dtoh_async(h_output, d_output, stream)
#     # Synchronize the stream
#     stream.synchronize()

#     assert h_output.shape == mge_result.shape
#     assert h_output.dtype == mge_result.dtype

#     np.testing.assert_allclose(h_output, mge_result, atol=max_err)
#     print("success!")



# @pytest.mark.parametrize("mode", ["normal","group"])
# def test_conv2d(mode):
#     net = ConvOpr(mode)
#     mge_result = dump_mge_model(net, net.data, tmp_file)
#     _test_convert_result(net.data, tmp_file, mge_result, max_error)



@pytest.mark.parametrize(
    "mode",
    [
        # "add",
        # "sub",
        # "mul",
        # "div",
        # "max",
        # "pow",
        "min",
        # "abs",
        # "exp",
        # "log",
        # "floor",
        # "ceil",
    ],
)
def test_elemwise(mode):
    net = ElemwiseOpr(mode)
    mge_result = dump_mge_model(net, net.data, tmp_file)
    _test_convert_result(net.data, tmp_file, mge_result, max_error)


# @pytest.mark.parametrize(
#     "mode",
#     [
#         # "relu",
#         # "relu6",        
#         "softmax",
#         # "sigmoid",
#         # "leaky_relu",
#         # "tanh",
#         # "hswish"
#     ],
# )
# def test_active(mode):
#     net = ActiveOpr(mode, fused=False)
#     mge_result = dump_mge_model(net, net.data, tmp_file)
#     _test_convert_result(net.data, tmp_file, mge_result, max_error)


# @pytest.mark.skipif(
#     mge.__version__ < "1.5.0",                                   
#     reason="MGE file for testing was dumped at version 1.5.0",
# )
# def test_flatten():
#     net = FlattenOpr()
#     mge_result = dump_mge_model(net, net.data, tmp_file)
#     _test_convert_result(net.data, tmp_file, mge_result, max_error)

# @pytest.mark.skipif(
#     mge.__version__ < "1.5.0",
#     reason="MGE file for testing was dumped at version 1.5.0",
# )
# def test_transopse():    
#     net = TransposeOpr()            
#     mge_result = dump_mge_model(net, net.data)
#     _test_convert_result(net.data, tmp_file, mge_result, max_error)

# @pytest.mark.skipif(
#     mge.__version__ < "1.5.0",
#     reason="MGE file for testing was dumped at version 1.5.0",
# )
# def test_reshape():
#     net = ReshapeOpr()
#     mge_result = dump_mge_model(net, net.data, tmp_file)
#     _test_convert_result(net.data, tmp_file, mge_result, max_error)

# @pytest.mark.parametrize(
#     "mode",
#     [
#         "sum",
#         "mean",
#         "max",
#         "min"
#     ],
# )
# def test_reduce(mode):
#     net = ReduceOpr(mode=mode)
#     mge_result = dump_mge_model(net, net.data, tmp_file)
#     _test_convert_result(net.data, tmp_file, mge_result, max_error)


# @pytest.mark.parametrize("mode", ["max","avg"])
# def test_pooling(mode):
#     net = PoolOpr(mode)
#     mge_result = dump_mge_model(net, net.data, tmp_file)
#     _test_convert_result(net.data, tmp_file, mge_result, max_error)


# @pytest.mark.skipif(
#     mge.__version__ < "1.5.0",
#     reason="MGE file for testing was dumped at version 1.5.0",
# )
# def test_linear():
#     net = LinearOpr()
#     mge_result = dump_mge_model(net, net.data, tmp_file)
#     _test_convert_result(net.data, tmp_file, mge_result, max_error)


# @pytest.mark.skipif(
#     mge.__version__ < "1.5.0",
#     reason="MGE file for testing was dumped at version 1.5.0",
# )
# def test_slice():
#     net = SubtensorOpr()
#     mge_result = dump_mge_model(net, net.data, tmp_file)
#     _test_convert_result(
#         net.data, tmp_file, mge_result, max_error
#     )

# @pytest.mark.skipif(
#     mge.__version__ < "1.5.0",
#     reason="MGE file for testing was dumped at version 1.5.0",
# )
# def test_slice():
#     net = ConcatOpr()
#     mge_result = dump_mge_model(net, net.data, tmp_file)
#     _test_convert_result(
#         net.data, tmp_file, mge_result, max_error
#     )

# @pytest.mark.skipif(
#     mge.__version__ < "1.5.0",
#     reason="MGE file for testing was dumped at version 1.5.0",
# )
# def test_adaptive():
#     net = AdaptiveAvgPool2dOpr()
#     net.eval()
#     data = net.data
#     mge_result = dump_mge_model(net, data, tmp_file)
#     _test_convert_result(data, tmp_file, mge_result, max_error)


# @pytest.mark.parametrize("mode", ["bn1d","bn2d"])
# def test_batchnorm(mode):
#     net = BnOpr(mode)
#     net.eval()
#     data = net.data1 if mode == "bn1d" else net.data2
#     mge_result = dump_mge_model(net, data, tmp_file)
#     _test_convert_result(data, tmp_file, mge_result, max_error)


# @pytest.mark.parametrize(
#     "model",
#     [
#         # "shufflenet_v2_x0_5",
#         # "shufflenet_v2_x1_0",
#         "resnet18",
#         # "resnet50",
#         # "resnet101",
#         # "resnext50_32x4d",
#     ],
# )
# def test_model(model):
#     data = (
#         np.random.randint(0, 255, 3 * 224 * 224)
#         .reshape((1, 3, 224, 224))
#         .astype(np.float32)
#     )
#     if megengine.__version__ < "1.1.0":
#         commit_id = "dc2f2cfb228a135747d083517b98aea56e7aab92"
#     else:
#         commit_id = None
#     net = megengine.hub.load(
#         "megengine/models", model, use_cache=False, commit=commit_id, pretrained=True
#     )
#     mge_result = dump_mge_model(net, data, tmp_file)
#     _test_convert_result(data, tmp_file, mge_result, 1e-2)


# @pytest.mark.skipif(
#     mge.__version__ < "1.5.0",
#     reason="MGE file for testing was dumped at version 1.5.0",
# )
# def test_model():
#     for model in get_public_cls_models():
#         model.eval()
#         data = (
#             np.random.randint(0, 255, 3 * 224 * 224)
#             .reshape((1, 3, 224, 224))
#             .astype(np.float32)
#         )
#         mge_result = dump_mge_model(model, data, tmp_file)
#         _test_convert_result(data, tmp_file, mge_result, 1e-2)


# @pytest.mark.skipif(
#     mge.__version__ < "1.5.0",
#     reason="MGE file for testing was dumped at version 1.5.0",
# )
# def test_model():
#     data = (
#         np.random.randint(0, 255, 3 * 224 * 224)
#         .reshape((1, 3, 224, 224))
#         .astype(np.float32)
#         # np.zeros((1, 3, 224, 224)).astype(np.float32)
#     )
    
#     net = se_resnet18()
#     mge_result = dump_mge_model(net, data, tmp_file)
#     _test_convert_result(data, tmp_file, mge_result, 1e-2)

#########################################################################################################3




from basecls.configs import (
    EffNetConfig,
    EffNetLiteConfig,
    MBConfig,
    RegNetConfig,
    ResNetConfig,
    SNetConfig,
    VGGConfig,
    ViTConfig,
)
from basecls.models import build_model

resnet_models = [
    "resnet101",  
    "resnet101d",  
    "resnet152",     
    "resnet152d",   
    "resnet18",
    "resnet18d",
    "resnet34",
    "resnet34d",
    "resnet50",
    "resnet50d",
    "resnext101_32x4d", 
    "resnext101_32x8d",
    "resnext101_64x4d",
    "resnext152_32x4d",
    "resnext152_32x8d",
    "resnext152_64x4d",
    "resnext50_32x4d",
    "se_resnet101",    ###
    "se_resnet152",    ###
    "se_resnet18",     
    "se_resnet34",     
    "se_resnet50",     
    "se_resnext101_32x4d",
    "se_resnext101_32x8d",
    "se_resnext101_64x4d",
    "se_resnext152_32x4d",
    "se_resnext152_32x8d",
    "se_resnext152_64x4d",
    "se_resnext50_32x4d",
    "wide_resnet101_2",
    "wide_resnet50_2",
]

regnet_models = [
    "regnetx_002",
    "regnetx_004",
    "regnetx_006",
    "regnetx_008",
    "regnetx_016",
    "regnetx_032",
    "regnetx_040",
    "regnetx_064",
    "regnetx_080",
    "regnetx_120",
    "regnetx_160",
    "regnetx_320",
    "regnety_002",
    "regnety_004",
    "regnety_006",
    "regnety_008",
    "regnety_016",
    "regnety_032",
    "regnety_040",
    "regnety_064",
    "regnety_080",
    "regnety_120",
    "regnety_160",
    "regnety_320",
]

vgg_models = [
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
]


effnet_models = [
    "effnet_b0",
    "effnet_b1",
    "effnet_b2",
    "effnet_b3",
    "effnet_b4",
    "effnet_b5",
    "effnet_b0_lite",
    "effnet_b1_lite",
    "effnet_b2_lite",
    "effnet_b3_lite",
    "effnet_b4_lite",
]

vit_models = [
    "vit_base_patch16_224",
    # "vit_base_patch16_384",
    # "vit_base_patch32_224",
    # "vit_base_patch32_384",
    # "vit_small_patch16_224",
    # "vit_small_patch16_384",
    # "vit_small_patch32_224",
    # "vit_small_patch32_384",
    # "vit_tiny_patch16_224",
    # "vit_tiny_patch16_384",
]

hrnet_models = [
    "hrnet_w18",
    # "hrnet_w18_small_v1",
    # "hrnet_w18_small_v2",
    # "hrnet_w30",
    # "hrnet_w32",
    # "hrnet_w40",
    # "hrnet_w44",
    # "hrnet_w48",
    # "hrnet_w64",
]

snet_models = ["snetv2_x050", "snetv2_x100", "snetv2_x150", "snetv2_x200"]

mb_models = [
    # "mbnetv1_x025",
    # "mbnetv1_x075",
    # "mbnetv1_x100",
    # "mbnetv2_x035",
    # "mbnetv2_x050",
    # "mbnetv2_x075",
    # "mbnetv2_x100",
    # "mbnetv2_x140",
    "mbnetv3_large_x075",    ####
    # "mbnetv3_large_x100",
    # "mbnetv3_small_x075",
    # "mbnetv3_small_x100",
]


_cls_configs = {
    "resnet": ResNetConfig,
    "resnext": ResNetConfig,
    "regnet": RegNetConfig,
    "effnet": EffNetConfig,
    "vgg": VGGConfig,
    "vit": ViTConfig,
    "snet": SNetConfig,
    "mbnet": MBConfig,
}


def generate_cls_model(name, **kwargs):
    _cfg = dict(model=dict(name=name,),)

    cfg_cls = None
    for k, v in _cls_configs.items():
        # if "effnet" in name and "lite" in name:
        #     cfg_cls = EffNetLiteConfig
        #     break
        if k in name:
            cfg_cls = v
            break

    assert cfg_cls, "can not find the model config"

    class Config(cfg_cls):
        def __init__(self, values_or_file=None, **kwargs):
            super().__init__(_cfg)
            self.merge(values_or_file, **kwargs)

    cfg = Config()

    print("generate the model {} and config is {}".format(name, cfg_cls))
    model = build_model(cfg)
    return model


class PublicClsModelIter:
    def __init__(self, models):
        self.models = models

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        for name in self.models:
            yield generate_cls_model(name)


def get_public_cls_models() -> PublicClsModelIter:
    public_cls_models = (
        hrnet_models
    )
    return PublicClsModelIter(public_cls_models)

###################################################################################################################################

from collections import OrderedDict
from functools import partial
from typing import Any, List, Mapping, Optional

import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
from numbers import Real
from basecls.layers import build_head, conv2d, init_weights, norm2d
from basecls.layers.activations import activation
from basecls.utils import recursive_update, registers
from basecls.layers import (
    SE,
    DropPath,
    activation,
    build_head,
    conv2d,
    init_weights,
    make_divisible,
    norm2d,
    pool2d,
)

#####################################################################################################################3
from basecls.layers import DropPath
from megengine.utils.tuple_function import _pair as to_2tuple

import functools
import itertools
import math
import warnings
from typing import List, Sequence, Tuple, Union

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F
import megengine.module as M
import numpy as np
from basecore.config import ConfigDict
from basecore.utils import all_reduce
from megengine.module.batchnorm import _BatchNorm
from megengine.utils.module_stats import register_flops

from basecls.data import DataLoaderType


NORM_TYPES = (_BatchNorm, M.GroupNorm, M.InstanceNorm, M.LayerNorm)


class Preprocess(M.Module):
    def __init__(self, mean: Union[float, Sequence[float]], std: Union[float, Sequence[float]]):
        super().__init__()
        self.mean = mge.Tensor(np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1))
        self.std = mge.Tensor(np.array(std, dtype=np.float32).reshape(1, -1, 1, 1))

    def forward(self, inputs: Sequence[np.ndarray]) -> Tuple[mge.Tensor, mge.Tensor]:
        samples, targets = [mge.Tensor(x) for x in inputs]
        samples = (samples - self.mean) / self.std
        return samples, targets


def adjust_block_compatibility(
    ws: Sequence[int], bs: Sequence[float], gs: Sequence[int]
) -> Tuple[List[int], ...]:
    """Adjusts the compatibility of widths, bottlenecks and groups.

    Args:
        ws: widths.
        bs: bottleneck multipliers.
        gs: group widths.

    Returns:
        The adjusted widths, bottlenecks and groups.
    """
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    assert all(b < 1 or b % 1 == 0 for b in bs)
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs


def calculate_fan_in_and_fan_out(tensor: mge.Tensor, pytorch_style: bool = False):
    """Fixed :py:func:`megengine.module.init.calculate_fan_in_and_fan_out` for group conv2d.

    Note:
        The group conv2d kernel shape in MegEngine is ``(G, O/G, I/G, K, K)``. This function
        calculates ``fan_out = O/G * K * K`` as default, but PyTorch uses ``fan_out = O * K * K``.

    Args:
        tensor: tensor to be initialized.
        pytorch_style: utilize pytorch style init for group conv. Default: ``False``
    """
    if len(tensor.shape) not in (2, 4, 5):
        raise ValueError(
            "fan_in and fan_out can only be computed for tensor with 2/4/5 " "dimensions"
        )
    if len(tensor.shape) == 5:
        # `GOIKK` to `OIKK`
        tensor = tensor.reshape(-1, *tensor.shape[2:]) if pytorch_style else tensor[0]

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if len(tensor.shape) > 2:
        receptive_field_size = functools.reduce(lambda x, y: x * y, tensor.shape[2:], 1)
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def compute_precise_bn_stats(cfg: ConfigDict, model: M.Module, dataloader: DataLoaderType):
    """Computes precise BN stats on training data.

    Args:
        cfg: config for precising BN.
        model: model for precising BN.
        dataloader: dataloader for precising BN.
    """
    # Prepare the preprocessor
    preprocess = Preprocess(cfg.preprocess.img_mean, cfg.preprocess.img_std)
    # Compute the number of minibatches to use
    num_iter = int(cfg.bn.num_samples_precise / cfg.batch_size / dist.get_world_size())
    num_iter = min(num_iter, len(dataloader))
    # Retrieve the BN layers
    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    # Initialize BN stats storage for computing mean(mean(batch)) and mean(var(batch))
    running_means = [F.zeros_like(bn.running_mean) for bn in bns]
    running_vars = [F.zeros_like(bn.running_var) for bn in bns]
    # Remember momentum values
    momentums = [bn.momentum for bn in bns]
    # Set momentum to 0.0 to compute BN stats that only reflect the current batch
    for bn in bns:
        bn.momentum = 0.0
    # Average the BN stats for each BN layer over the batches
    for data in itertools.islice(dataloader, num_iter):
        samples, _ = preprocess(data)
        model(samples)
        for i, bn in enumerate(bns):
            running_means[i] += bn.running_mean / num_iter
            running_vars[i] += bn.running_var / num_iter
    # Sync BN stats across GPUs (no reduction if 1 GPU used)
    running_means = [all_reduce(x, mode="mean") for x in running_means]
    running_vars = [all_reduce(x, mode="mean") for x in running_vars]
    # Set BN stats and restore original momentum values
    for i, bn in enumerate(bns):
        bn.running_mean = running_means[i]
        bn.running_var = running_vars[i]
        bn.momentum = momentums[i]


def init_weights(m: M.Module, pytorch_style: bool = False, zero_init_final_gamma: bool = False):
    """Performs ResNet-style weight initialization.

    About zero-initialize:
    Zero-initialize the last BN in each residual branch, so that the residual branch starts
    with zeros, and each residual block behaves like an identity.
    This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677.

    Args:
        m: module to be initialized.
        pytorch_style: utilize pytorch style init for group conv. Default: ``False``
        zero_init_final_gamma: enable zero-initialize or not. Default: ``False``
    """

    if isinstance(m, M.Conv2d):
        _, fan_out = calculate_fan_in_and_fan_out(m.weight, pytorch_style)
        std = math.sqrt(2 / fan_out)
        M.init.normal_(m.weight, 0, std)
        if getattr(m, "bias", None) is not None:
            fan_in, _ = calculate_fan_in_and_fan_out(m.weight, pytorch_style)
            bound = 1 / math.sqrt(fan_in)
            M.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, NORM_TYPES):
        M.init.fill_(
            m.weight, 0.0 if getattr(m, "final_bn", False) and zero_init_final_gamma else 1.0
        )
        M.init.zeros_(m.bias)
    elif isinstance(m, M.Linear):
        M.init.normal_(m.weight, std=0.01)
        if getattr(m, "bias", None) is not None:
            M.init.zeros_(m.bias)


def init_vit_weights(module: M.Module):
    if isinstance(module, M.Linear):
        if module.name and module.name.startswith("head"):
            M.init.zeros_(module.weight)
            M.init.zeros_(module.bias)
        elif module.name and module.name.startswith("pre_logits"):
            lecun_normal_(module.weight)
            M.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                M.init.zeros_(module.bias)
    elif isinstance(module, M.Conv2d):
        M.init.msra_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = M.init.calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            M.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, (M.LayerNorm, M.GroupNorm, M.BatchNorm2d)):
        M.init.zeros_(module.bias)
        M.init.ones_(module.weight)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    lo = norm_cdf((a - mean) / std)
    up = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    M.init.uniform_(tensor, 2 * lo - 1, 2 * up - 1)
    # tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor._reset(M.Elemwise("erfinv")(tensor))
    # tensor.erfinv_()

    # Transform to proper mean, std
    tensor *= std * math.sqrt(2.0)
    # tensor.mul_(std * math.sqrt(2.))
    tensor += mean
    # tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor._reset(F.clip(tensor, lower=a, upper=b))
    # tensor.clamp_(min=a, max=b)
    return tensor


def lecun_normal_(tensor):
    fan_in, _ = calculate_fan_in_and_fan_out(tensor)
    std = 1 / math.sqrt(fan_in) / 0.87962566103423978
    # constant is stddev of standard normal truncated to (-2, 2)
    trunc_normal_(tensor, std=std)


def make_divisible(value, divisor: int = 8, min_value: int = None, round_limit: float = 0.0):
    min_value = min_value or divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < round_limit * value:
        new_value += divisor
    return new_value


class FFN(M.Module):
    """FFN for ViT

    Args:
        in_features: Number of input features.
        hidden_features: Number of input features. Default: ``None``
        out_features: Number of output features. Default: ``None``
        act_name: activation function. Default: ``"gelu"``
        drop: Dropout ratio. Default: ``0.0``
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_name: str = "gelu",
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = M.Linear(in_features, hidden_features)
        self.act = activation(act_name)
        self.fc2 = M.Linear(hidden_features, out_features)
        self.drop = M.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(M.Module):
    """Image to Patch Embedding

    Args:
        img_size: Image size.  Default: ``224``
        patch_size: Patch token size. Default: ``16``
        in_chans: Number of input image channels. Default: ``3``
        embed_dim: Number of linear projection output channels. Default: ``768``
        flatten: Flatten embedding. Default: ``True``
        norm_name: Normalization layer. Default: ``None``
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
        norm_name: str = None,
        **kwargs,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = M.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm2d(norm_name, embed_dim) if norm_name else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        if self.flatten:
            x = F.flatten(x, 2).transpose(0, 2, 1)
        if self.norm:
            x = self.norm(x)
        return x




class Affine(M.Module):
    """ResMLP Affine Layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.alpha = mge.Parameter(F.ones(dim))
        self.beta = mge.Parameter(F.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta


class ResMLPBlock(M.Module):
    """ResMLP block.

    Args:
        dim: Number of input channels.
        drop: Dropout ratio.
        drop_path: Stochastic depth rate.
        num_patches: Number of patches.
        init_scale: Initial value for LayerScale.
        ffn_ratio: Ratio of ffn hidden dim to embedding dim.
        act_name: activation function.
    """

    def __init__(
        self,
        dim: int,
        drop: float,
        drop_path: float,
        num_patches: int,
        init_scale: float,
        ffn_ratio: float,
        act_name: str,
    ):
        super().__init__()
        self.norm1 = Affine(dim)
        self.attn = M.Linear(num_patches, num_patches)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None
        self.norm2 = Affine(dim)
        self.mlp = FFN(
            in_features=dim, hidden_features=int(ffn_ratio * dim), act_name=act_name, drop=drop
        )
        self.gamma1 = mge.Parameter(init_scale * F.ones((dim)))
        self.gamma2 = mge.Parameter(init_scale * F.ones((dim)))

    def forward(self, x):
        if self.drop_path:
            x = x + self.drop_path(
                self.gamma1 * self.attn(self.norm1(x).transpose(0, 2, 1)).transpose(0, 2, 1)
            )
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.gamma1 * self.attn(self.norm1(x).transpose(0, 2, 1)).transpose(0, 2, 1)
            x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


@registers.models.register()
class ResMLP(M.Module):
    """ResMLP model.

    Args:
        img_size: Input image size. Default: ``224``
        patch_size: Patch token size. Default: ``16``
        in_chans: Number of input image channels. Default: ``3``
        embed_dim: Number of linear projection output channels. Default: ``768``
        depth: Depth of Transformer Encoder layer. Default: ``12``
        drop_rate: Dropout rate. Default: ``0.0``
        drop_path_rate: Stochastic depth rate. Default: ``0.0``
        embed_layer: Patch embedding layer. Default: :py:class:`PatchEmbed`
        init_scale: Initial value for LayerScale. Default: ``1e-4``
        ffn_ratio: Ratio of ffn hidden dim to embedding dim. Default: ``4.0``
        act_name: Activation function. Default: ``"gelu"``
        num_classes: Number of classes. Default: ``1000``
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: M.Module = PatchEmbed,
        init_scale: float = 1e-4,
        ffn_ratio: float = 4.0,
        act_name: str = "gelu",
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = embed_layer(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        dpr = [drop_path_rate for _ in range(depth)]

        self.blocks = [
            ResMLPBlock(
                dim=embed_dim,
                drop=drop_rate,
                drop_path=dpr[i],
                num_patches=num_patches,
                init_scale=init_scale,
                ffn_ratio=ffn_ratio,
                act_name=act_name,
            )
            for i in range(depth)
        ]
        self.norm = Affine(embed_dim)
        self.head = M.Linear(embed_dim, num_classes) if num_classes > 0 else None

        self.apply(init_vit_weights)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(axis=1).reshape(B, 1, -1)
        x = x[:, 0]
        if self.head:
            x = self.head(x)
        return x


def _build_resmlp(**kwargs):
    model_args = dict(embed_dim=384, drop_path_rate=0.05)
    recursive_update(model_args, kwargs)
    return ResMLP(**model_args)

def resmlp_s12(**kwargs):
    model_args = dict(depth=12, init_scale=0.1)
    recursive_update(model_args, kwargs)
    return _build_resmlp(**model_args)

def resmlp_s24(**kwargs):
    model_args = dict(depth=24, init_scale=1e-5)
    recursive_update(model_args, kwargs)
    return _build_resmlp(**model_args)

def resmlp_s36(**kwargs):
    model_args = dict(depth=36, init_scale=1e-6)
    recursive_update(model_args, kwargs)
    return _build_resmlp(**model_args)

def resmlp_b24(**kwargs):
    model_args = dict(patch_size=8, embed_dim=768, depth=24, init_scale=1e-6)
    recursive_update(model_args, kwargs)
    return _build_resmlp(**model_args)


###################################################################################################################################3
from functools import partial
from numbers import Real
from typing import Any, Callable, Mapping, Sequence, Union


class ResBasicBlock(M.Module):
    """Residual basic block: x + f(x), f = [3x3 conv, BN, Act] x2."""

    def __init__(
        self,
        w_in: int,
        w_out: int,
        stride: int,
        bot_mul: float,
        se_r: float,
        avg_down: bool,
        norm_name: str,
        act_name: str,
        **kwargs,
    ):
        super().__init__()
        if w_in != w_out or stride > 1:
            if avg_down and stride > 1:
                self.pool = M.AvgPool2d(2, stride)
                self.proj = conv2d(w_in, w_out, 1)
            else:
                self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(norm_name, w_out)

        w_b = round(w_out * bot_mul)
        w_se = make_divisible(w_out * se_r) if se_r > 0.0 else 0
        self.a = conv2d(w_in, w_b, 3, stride=stride)
        self.a_bn = norm2d(norm_name, w_b)
        self.a_act = activation(act_name)
        self.b = conv2d(w_b, w_out, 3)
        self.b_bn = norm2d(norm_name, w_out)
        self.b_bn.final_bn = True
        if w_se > 0:
            self.se = SE(w_out, w_se, act_name)
        self.act = activation(act_name)

    def forward(self, x):
        x_p = x
        if getattr(self, "pool", None) is not None:
            x_p = self.pool(x_p)
        if getattr(self, "proj", None) is not None:
            x_p = self.proj(x_p)
            x_p = self.bn(x_p)

        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_act(x)
        x = self.b(x)
        x = self.b_bn(x)
        if getattr(self, "se", None) is not None:
            x = self.se(x)
        x += x_p
        x = self.act(x)
        return x


class ResBottleneckBlock(M.Module):
    """Residual bottleneck block: x + f(x), f = 1x1, 3x3, 1x1 [+SE]."""

    def __init__(
        self,
        w_in: int,
        w_out: int,
        stride: int,
        bot_mul: float,
        group_w: int,
        se_r: float,
        avg_down: bool,
        norm_name: str,
        act_name: str,
        **kwargs,
    ):
        super().__init__()
        if w_in != w_out or stride > 1:
            if avg_down and stride > 1:
                self.pool = M.AvgPool2d(2, stride)
                self.proj = conv2d(w_in, w_out, 1)
            else:
                self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(norm_name, w_out)

        w_b = round(w_out * bot_mul)
        w_se = make_divisible(w_out * se_r) if se_r > 0.0 else 0
        groups = w_b // group_w
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(norm_name, w_b)
        self.a_act = activation(act_name)
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(norm_name, w_b)
        self.b_act = activation(act_name)
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(norm_name, w_out)
        self.c_bn.final_bn = True
        if w_se > 0:
            self.se = SE(w_out, w_se, act_name)
        self.act = activation(act_name)

    def forward(self, x):
        x_p = x
        if getattr(self, "pool", None) is not None:
            x_p = self.pool(x_p)
        if getattr(self, "proj", None) is not None:
            x_p = self.proj(x_p)
            x_p = self.bn(x_p)

        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_act(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_act(x)
        x = self.c(x)
        x = self.c_bn(x)
        if getattr(self, "se", None) is not None:
            x = self.se(x)
        x += x_p
        # x = self.act(x)
        return x

class ResBottleneckBlock9(M.Module):
    """Residual bottleneck block: x + f(x), f = 1x1, 3x3, 1x1 [+SE]."""

    def __init__(
        self,
        w_in: int,
        w_out: int,
        stride: int,
        bot_mul: float,
        group_w: int,
        se_r: float,
        avg_down: bool,
        norm_name: str,
        act_name: str,
        **kwargs,
    ):
        super().__init__()
        if w_in != w_out or stride > 1:
            if avg_down and stride > 1:
                self.pool = M.AvgPool2d(2, stride)
                self.proj = conv2d(w_in, w_out, 1)
            else:
                self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(norm_name, w_out)

        w_b = round(w_out * bot_mul)
        w_se = make_divisible(w_out * se_r) if se_r > 0.0 else 0
        groups = w_b // group_w
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(norm_name, w_b)
        self.a_act = activation(act_name)
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(norm_name, w_b)
        self.b_act = activation(act_name)
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(norm_name, w_out)
        self.c_bn.final_bn = True
        if w_se > 0:
            self.se = SE(w_out, w_se, act_name)
        self.act = activation(act_name)

    def forward(self, x):
        # x_p = x
        # if getattr(self, "pool", None) is not None:
        #     x_p = self.pool(x_p)
        # if getattr(self, "proj", None) is not None:
        #     x_p = self.proj(x_p)
        #     x_p = self.bn(x_p)

        # x = self.a(x)
        # x = self.a_bn(x)
        # x = self.a_act(x)
        # x = self.b(x)
        # x = self.b_bn(x)
        # x = self.b_act(x)
        # x = self.c(x)
        # x = self.c_bn(x)
        # if getattr(self, "se", None) is not None:
        #     x = self.se(x)
        # x += x_p
        # x = self.act(x)
        return x


class ResDeepStem(M.Module):
    """ResNet-D stem: [3x3, BN, Act] x3, MaxPool."""

    def __init__(self, w_in: int, w_out: int, norm_name: str, act_name: str, **kwargs):
        super().__init__()
        w_b = w_out // 2
        self.a = conv2d(w_in, w_b, 3, stride=2)
        self.a_bn = norm2d(norm_name, w_b)
        self.a_act = activation(act_name)
        self.b = conv2d(w_b, w_b, 3, stride=1)
        self.b_bn = norm2d(norm_name, w_b)
        self.b_act = activation(act_name)
        self.c = conv2d(w_b, w_out, 3, stride=1)
        self.c_bn = norm2d(norm_name, w_out)
        self.c_act = activation(act_name)
        self.pool = pool2d(3, stride=2)

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_act(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_act(x)
        x = self.c(x)
        x = self.c_bn(x)
        x = self.c_act(x)
        x = self.pool(x)
        return x


class ResStem(M.Module):
    """ResNet stem: 7x7, BN, Act, MaxPool."""

    def __init__(self, w_in: int, w_out: int, norm_name: str, act_name: str, **kwargs):
        super().__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(norm_name, w_out)
        self.act = activation(act_name)
        self.pool = pool2d(3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class SimpleStem(M.Module):
    """Simple stem: 3x3, BN, Act."""

    def __init__(self, w_in: int, w_out: int, norm_name: str, act_name: str, **kwargs):
        super().__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(norm_name, w_out)
        self.act = activation(act_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class AnyStage(M.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self, w_in: int, w_out: int, stride: int, depth: int, block_func: Callable, **kwargs
    ):
        super().__init__()
        self.depth = depth
        for i in range(depth):
            block = block_func(w_in, w_out, stride, **kwargs)
            setattr(self, f"b{i + 1}", block)
            stride, w_in = 1, w_out

    def __len__(self):
        return self.depth

    def forward(self, x):
        for i in range(self.depth):
            block = getattr(self, f"b{i + 1}")
            x = block(x)
        return x

class AnyStage3(M.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self, w_in: int, w_out: int, stride: int, depth: int, block_func: Callable, **kwargs
    ):
        super().__init__()
        self.depth = depth
        for i in range(23):
            block = block_func(w_in, w_out, stride, **kwargs)  # 1024 1024 1
            setattr(self, f"b{i + 1}", block)
            stride, w_in = 1, w_out

        # block1 = ResBottleneckBlock9(1024, 1024, 1, **kwargs)  # 1024 1024 1
        # setattr(self, f"b{23}", block1)

    def __len__(self):
        return self.depth

    def forward(self, x):
        for i in range(23):
            block = getattr(self, f"b{i + 1}")
            x = block(x)
        return x


class ResNet(M.Module):
    """ResNet model.

    Args:
        stem_name: stem name.
        stem_w: stem width.
        block_name: block name.
        depths: depth for each stage (number of blocks in the stage).
        widths: width for each stage (width of each block in the stage).
        strides: strides for each stage (applies to the first block of each stage).
        bot_muls: bottleneck multipliers for each stage (applies to bottleneck block).
            Default: ``1.0``
        group_ws: group widths for each stage (applies to bottleneck block). Default: ``None``
        se_r: Squeeze-and-Excitation (SE) ratio. Default: ``0.0``
        zero_init_final_gamma: enable zero-initialize or not. Default: ``False``
        norm_name: normalization function. Default: ``"BN"``
        act_name: activation function. Default: ``"relu"``
        head: head args. Default: ``None``
    """

    def __init__(
        self,
        stem_name: Union[str, Callable],   #  "ResStem"
        stem_w: int,                       #  64
        block_name: Union[str, Callable],  #  "ResBottleneckBlock"
        depths: Sequence[int],             #  [3, 4, 23, 3]
        widths: Sequence[int],             #  [256, 512, 1024, 2048]
        strides: Sequence[int],            #  [1, 2, 2, 2]
        bot_muls: Union[float, Sequence[float]] = 1.0, # [0.25, 0.25, 0.25, 0.25]
        group_ws: Sequence[int] = None,    # [64, 128, 256, 512]
        se_r: float = 0.0,                 # 0.0625
        avg_down: bool = False,            
        zero_init_final_gamma: bool = False,
        norm_name: str = "BN",
        act_name: str = "relu",
        head: Mapping[str, Any] = None,    # {'name': 'ClsHead'}
    ):
        super().__init__()
        self.depths = depths
        stem_func = self.get_stem_func(stem_name)
        self.stem = stem_func(3, stem_w, norm_name, act_name)

        block_func = self.get_block_func(block_name)
        if isinstance(bot_muls, Real):
            bot_muls = [bot_muls] * len(depths)
        if group_ws is None:
            group_ws = [None] * len(depths)
        model_args = [depths, widths, strides, bot_muls, group_ws]
        prev_w = stem_w

        for i, (d, w, s, b, g) in enumerate(zip(*model_args)):
            if i == 2:
                break
            stage = AnyStage(
                prev_w,  ## 512
                w,       ## 1024
                s,       #  2
                d,       #  23
                block_func, #   ResBottleneckBlock
                bot_mul=b,  #   0.25
                group_w=g,  #   256
                se_r=se_r,   # 0.0625
                avg_down=avg_down, # False
                norm_name=norm_name, # "BN"
                act_name=act_name,  # "relu"
            )
            setattr(self, f"s{i + 1}", stage)
            prev_w = w

        stage3 = AnyStage3(
                512,  ## 512
                1024,       ## 1024
                2,       #  2
                23,       #  23
                ResBottleneckBlock, #   ResBottleneckBlock
                bot_mul=0.25,  #   0.25
                group_w=256,  #   256
                se_r=0.0625,   # 0.0625
                avg_down=False, # False
                norm_name="BN", # "BN"
                act_name="relu",  # "relu"
            )
        setattr(self, f"s{3}", stage3)

        self.head = build_head(prev_w, head, norm_name, act_name)

        self.apply(
            partial(init_weights, pytorch_style=True, zero_init_final_gamma=zero_init_final_gamma)
        )

    def forward(self, x):
        x = self.stem(x)
        for i in range(3):
            stage = getattr(self, f"s{i + 1}")
            x = stage(x)
        # if getattr(self, "head", None) is not None:
        #     x = self.head(x)
        return x

    @staticmethod
    def get_stem_func(name: Union[str, Callable]):
        """Retrieves the stem function by name."""
        if callable(name):
            return name
        if isinstance(name, str):
            stem_funcs = {
                "ResDeepStem": ResDeepStem,
                "ResStem": ResStem,
                "SimpleStem": SimpleStem,
            }
            if name in stem_funcs.keys():
                return stem_funcs[name]
        raise ValueError(f"Stem '{name}' not supported")

    @staticmethod
    def get_block_func(name: Union[str, Callable]):
        """Retrieves the block function by name."""
        if callable(name):
            return name
        if isinstance(name, str):
            block_funcs = {
                "ResBasicBlock": ResBasicBlock,
                "ResBottleneckBlock": ResBottleneckBlock,
            }
            if name in block_funcs.keys():
                return block_funcs[name]
        raise ValueError(f"Block '{name}' not supported")


def _build_resnet(**kwargs):
    model_args = dict(stem_name=ResStem, stem_w=64, head=dict(name="ClsHead"))
    recursive_update(model_args, kwargs)
    return ResNet(**model_args)

def resnet18(**kwargs):
    model_args = dict(
        block_name=ResBasicBlock,
        depths=[2, 2, 2, 2],
        widths=[64, 128, 256, 512],
        strides=[1, 2, 2, 2],
    )
    recursive_update(model_args, kwargs)
    return _build_resnet(**model_args)

def resnet34(**kwargs):
    model_args = dict(
        block_name=ResBasicBlock,
        depths=[3, 4, 6, 3],
        widths=[64, 128, 256, 512],
        strides=[1, 2, 2, 2],
    )
    recursive_update(model_args, kwargs)
    return _build_resnet(**model_args)

def resnet50(**kwargs):
    model_args = dict(
        block_name=ResBottleneckBlock,
        depths=[3, 4, 6, 3],
        widths=[256, 512, 1024, 2048],
        strides=[1, 2, 2, 2],
        bot_muls=[0.25, 0.25, 0.25, 0.25],
        group_ws=[64, 128, 256, 512],
    )
    recursive_update(model_args, kwargs)
    return _build_resnet(**model_args)

def resnet101(**kwargs):
    model_args = dict(
        block_name=ResBottleneckBlock,
        depths=[3, 4, 23, 3],
        widths=[256, 512, 1024, 2048],
        strides=[1, 2, 2, 2],
        bot_muls=[0.25, 0.25, 0.25, 0.25],
        group_ws=[64, 128, 256, 512],
    )
    recursive_update(model_args, kwargs)
    return _build_resnet(**model_args)

def resnet152(**kwargs):
    model_args = dict(
        block_name=ResBottleneckBlock,
        depths=[3, 8, 36, 3],
        widths=[256, 512, 1024, 2048],
        strides=[1, 2, 2, 2],
        bot_muls=[0.25, 0.25, 0.25, 0.25],
        group_ws=[64, 128, 256, 512],
    )
    recursive_update(model_args, kwargs)
    return _build_resnet(**model_args)

def resnet18d(**kwargs):
    model_args = dict(stem_name=ResDeepStem, avg_down=True)
    recursive_update(model_args, kwargs)
    return resnet18(**model_args)

def resnet34d(**kwargs):
    model_args = dict(stem_name=ResDeepStem, avg_down=True)
    recursive_update(model_args, kwargs)
    return resnet34(**model_args)

def resnet50d(**kwargs):
    model_args = dict(stem_name=ResDeepStem, avg_down=True)
    recursive_update(model_args, kwargs)
    return resnet50(**model_args)

def resnet101d(**kwargs):
    model_args = dict(stem_name=ResDeepStem, avg_down=True)
    recursive_update(model_args, kwargs)
    return resnet101(**model_args)

def resnet152d(**kwargs):
    model_args = dict(stem_name=ResDeepStem, avg_down=True)
    recursive_update(model_args, kwargs)
    return resnet152(**model_args)

def resnext50_32x4d(**kwargs):
    model_args = dict(bot_muls=[0.5, 0.5, 0.5, 0.5], group_ws=[4, 8, 16, 32])
    recursive_update(model_args, kwargs)
    return resnet50(**model_args)

def resnext101_32x4d(**kwargs):
    model_args = dict(bot_muls=[0.5, 0.5, 0.5, 0.5], group_ws=[4, 8, 16, 32])
    recursive_update(model_args, kwargs)
    return resnet101(**model_args)

def resnext101_32x8d(**kwargs):
    model_args = dict(bot_muls=[1.0, 1.0, 1.0, 1.0], group_ws=[8, 16, 32, 64])
    recursive_update(model_args, kwargs)
    return resnet101(**model_args)

def resnext101_64x4d(**kwargs):
    model_args = dict(bot_muls=[1.0, 1.0, 1.0, 1.0], group_ws=[4, 8, 16, 32])
    recursive_update(model_args, kwargs)
    return resnet101(**model_args)

def resnext152_32x4d(**kwargs):
    model_args = dict(bot_muls=[0.5, 0.5, 0.5, 0.5], group_ws=[4, 8, 16, 32])
    recursive_update(model_args, kwargs)
    return resnet152(**model_args)

def resnext152_32x8d(**kwargs):
    model_args = dict(bot_muls=[1.0, 1.0, 1.0, 1.0], group_ws=[8, 16, 32, 64])
    recursive_update(model_args, kwargs)
    return resnet152(**model_args)

def resnext152_64x4d(**kwargs):
    model_args = dict(bot_muls=[1.0, 1.0, 1.0, 1.0], group_ws=[4, 8, 16, 32])
    recursive_update(model_args, kwargs)
    return resnet152(**model_args)

def se_resnet18(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnet18(**model_args)

def se_resnet34(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnet34(**model_args)

def se_resnet50(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnet50(**model_args)

def se_resnet101(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnet101(**model_args)

def se_resnet152(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnet152(**model_args)

def se_resnext50_32x4d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext50_32x4d(**model_args)

def se_resnext101_32x4d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext101_32x4d(**model_args)

def se_resnext101_32x8d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext101_32x8d(**model_args)

def se_resnext101_64x4d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext101_64x4d(**model_args)

def se_resnext152_32x4d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext152_32x4d(**model_args)

def se_resnext152_32x8d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext152_32x8d(**model_args)

def se_resnext152_64x4d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext152_64x4d(**model_args)

def wide_resnet50_2(**kwargs):
    model_args = dict(bot_muls=[0.5, 0.5, 0.5, 0.5])
    recursive_update(model_args, kwargs)
    return resnet50(**model_args)

def wide_resnet101_2(**kwargs):
    model_args = dict(bot_muls=[0.5, 0.5, 0.5, 0.5])
    recursive_update(model_args, kwargs)
    return resnet101(**model_args)
