# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=no-member

import random

import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
from megengine.jit import trace


def dump_mge_model(net, data, fpath="test_model", optimize_for_inference=False):
    if mge.__version__ <= "0.6.0":

        @trace(symbolic=True)
        def inference(data, *, net):
            net.eval()
            output = net(data)
            return output

        inference.trace(data, net=net)
        mge_result = inference(data, net=net).numpy()
        inference.dump(
            fpath + ".mge",
            arg_names=["data"],
            optimize_for_inference=optimize_for_inference,
        )
        return mge_result
    else:
        mge_result = net(mge.tensor(data))
        net.eval()
        mge_result = net(mge.tensor(data))

        @trace(symbolic=True, capture_as_const=True)
        def inference(data):
            net.eval()
            output = net(data)
            return output

        inference(mge.tensor(data))
        inference.dump(
            fpath + ".mge",
            arg_names=["data"],
            optimize_for_inference=optimize_for_inference,
        )
        return mge_result.numpy()


class ConvOpr(M.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.data = np.random.random((1, 3, 224, 224)).astype(np.float32)
        self.normal_conv = M.Conv2d(
            3, 30, 3, stride=(2, 3), dilation=(2, 2), padding=(3, 1)
        )
        self.group_conv = M.Conv2d(
            3, 30, 3, stride=(2, 3), dilation=(2, 2), padding=(3, 1), groups=3
        )

        self.valid_pad_conv = M.Conv2d(3, 30, 4, padding=(1, 1))
        self.valid_pad_1_conv = M.Conv2d(3, 30, 3, stride=2, padding=(1, 1))
        self.same_pad_conv = M.Conv2d(3, 30, 3, padding=(1, 1))
        self.same_pad_1_conv = M.Conv2d(3, 30, 4, stride=2, padding=(1, 1))
        self.same_pad_2_conv = M.Conv2d(3, 30, 2, dilation=3, stride=2, padding=(1, 1))

        self.normal_conv.bias = mge.Parameter(
            np.random.random(self.normal_conv.bias.shape).astype(np.float32)
        )
        self.group_conv.bias = mge.Parameter(
            np.random.random(self.group_conv.bias.shape).astype(np.float32)
        )

        self.transpose_conv = M.Sequential(
            M.ConvTranspose2d(
                3, 5, (3, 4), dilation=(2, 2), stride=(3, 2), padding=(2, 3), groups=1
            ),
            M.ConvTranspose2d(5, 3, (3, 3)),
        )
        self.transpose_conv[0].bias = mge.Parameter(
            np.random.random(self.transpose_conv[0].bias.shape).astype(np.float32)
        )
        self.transpose_conv[1].bias = mge.Parameter(
            np.random.random(self.transpose_conv[1].bias.shape).astype(np.float32)
        )

        self.tflite_transpose_conv = M.Sequential(
            M.ConvTranspose2d(3, 5, (3, 4), stride=(3, 2), groups=1),
            M.ConvTranspose2d(5, 3, (3, 3)),
        )
        self.tflite_transpose_conv[0].bias = mge.Parameter(
            np.random.random(self.transpose_conv[0].bias.shape).astype(np.float32)
        )
        self.tflite_transpose_conv[1].bias = mge.Parameter(
            np.random.random(self.transpose_conv[1].bias.shape).astype(np.float32)
        )

    def forward(self, x):
        return getattr(self, self.mode + "_conv")(x)


class ConvRelu2dOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((1, 3, 224, 224)).astype(np.float32)
        self.convrelu = M.ConvRelu2d(
            3, 10, 3, stride=(2, 3), dilation=(2, 2), padding=(3, 1)
        )

    def forward(self, x):
        return self.convrelu(x)


class LinearOpr(M.Module):
    def __init__(self, mode="normal"):
        super().__init__()
        self.data = np.random.random((10, 100)).astype(np.float32)
        self.data1 = np.random.random((10, 10, 10)).astype(np.float32)
        self.linear = M.Linear(100, 200, bias=False)
        self.linear_bias = M.Linear(200, 200, bias=True)
        self.linear_bias.bias = mge.Parameter(
            np.random.random(self.linear_bias.bias.shape).astype(np.float32)
        )
        self.mode = mode

    def forward(self, x):
        if self.mode == "flatten":
            x = F.flatten(x, 1)

        x = self.linear(x)
        x = self.linear_bias(x)
        return x


class LinearBnOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((10, 100)).astype(np.float32)
        self.linear = M.Linear(100, 200, bias=False)
        self.bn = M.BatchNorm1d(200)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return x


class MatrixMulBnOpr(M.Module):
    def __init__(self, transpose=False):
        super().__init__()
        self.transpose = transpose
        self.data = np.random.random((10, 100)).astype(np.float32)
        weight = np.random.random((200, 100) if transpose else (100, 200))
        self.linear_weight = mge.Tensor(weight)
        self.bn = M.BatchNorm1d(200)

    def forward(self, x):
        x = F.matmul(x, self.linear_weight, transpose_b=self.transpose)
        x = self.bn(x)
        return x


class PoolOpr(M.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.data = np.random.random((30, 3, 224, 224)).astype(np.float32)
        self.maxpool = M.pooling.MaxPool2d(kernel_size=3, stride=2, padding=2)
        self.avgpool = M.pooling.AvgPool2d(kernel_size=3, stride=2, padding=2)

    def forward(self, x):
        return getattr(self, self.mode + "pool")(x)


class PadOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((3, 3, 64, 64)).astype(np.float32)
        self.pad = M.Pad(((0, 0), (0, 0), (2, 2), (3, 3),))

    def forward(self, x):
        x = F.nn.pad(self.pad(x), ((0, 0), (0, 0), (2, 2), (3, 3),))
        return x


class BnOpr(M.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.data1 = np.random.random((1, 32, 32)).astype(np.float32)
        self.data2 = np.random.random((20, 3, 24, 24)).astype(np.float32)
        self.bn1d = M.BatchNorm1d(32)
        self.bn2d = M.BatchNorm2d(3)

    def forward(self, x):
        return getattr(self, self.mode)(x)


class ConvBn2dOpr(M.Module):
    def __init__(self, has_bias=True):
        super().__init__()
        self.data = np.random.random((1, 3, 224, 224)).astype(np.float32)
        self.convbn = M.ConvBn2d(
            3, 10, 3, stride=(2, 3), dilation=(2, 2), padding=(3, 1), bias=has_bias
        )

    def forward(self, x):
        return self.convbn(x)


class ConvBnRelu2dOpr(M.Module):
    def __init__(self, has_bias=True):
        super().__init__()
        self.data = np.random.random((1, 3, 224, 224)).astype(np.float32)
        self.convbnrelu = M.ConvBnRelu2d(
            3, 10, 3, stride=(2, 3), dilation=(2, 2), padding=(3, 1), bias=has_bias
        )

    def forward(self, x):
        return self.convbnrelu(x)


class SubtensorOpr(M.Module):
    def __init__(self, fix_batch=False):
        super().__init__()
        self.fix_batch = fix_batch
        self.data = np.random.random((10, 10, 10, 10)).astype(np.float32)

    def forward(self, x):
        if self.fix_batch:
            x = x[:, 4:8, :, 4:9]
            x = x[:, :, 2:7, 3]
        else:
            x = x[1:3, 4:8, :, 4:9]
            x = x[:, :, :, 3]
            x = x[1, 1:]
        return x


class NCHW_SubtensorOpr(M.Module):
    def __init__(self, fix_batch=False):
        super().__init__()
        self.fix_batch = fix_batch
        self.data = np.random.random((5, 10, 20, 20)).astype(np.float32)

    def forward(self, x):
        if self.fix_batch:
            x = x[:, 4:8, 4:9, 2]
        else:
            x = x[1:3, 4:8, 4:9, 2]
        return x


class TransposeOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((1, 2, 3, 4)).astype(np.float32)
        self.perm = [0, 2, 3, 1]

    def forward(self, x):
        return F.transpose(x, self.perm)


class ConcatOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.concat_idx = random.randint(0, 3)
        self.data = np.random.random((1, 2, 4, 5)).astype(np.float32)

    def forward(self, a):
        return F.concat([a, a], self.concat_idx)


class FConcatOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.concat_idx = random.randint(0, 3)

    def forward(self, inps):
        return F.concat(inps, self.concat_idx)


class MConcatOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.concat_idx = random.randint(0, 3)
        self.m = M.Concat()

    def forward(self, inps, axis=0):
        return self.m(inps, axis)


class SoftmaxOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((1, 1000)).astype(np.float32)

    def forward(self, a):
        return F.softmax(a)


class SqueezeOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((1, 1, 1000)).astype(np.float32)

    def forward(self, a):
        if mge.__version__ <= "0.6.0":
            return F.remove_axis(a, 0)  # pylint: disable=no-member
        else:
            return F.squeeze(a, 0)


class ReshapeOpr(M.Module):
    def __init__(self, fix_batch=False):
        super().__init__()
        if fix_batch:
            self.data = np.random.random((1, 2, 3, 4)).astype(np.float32)
            self.out_shape = (1, 2 * 3, 4)
            self.out_shape1 = (1, 2 * 3 * 4)
            self.out_shape2 = (1, 2, 3 * 4)
        else:
            self.data = np.random.random((1, 2, 3, 4, 5)).astype(np.float32)
            self.out_shape = [1, 2, 3 * 4, 5]
            self.out_shape1 = [1 * 2, 3 * 4 * 5]
            self.out_shape2 = [1 * 2 * 3, 4 * 5]

    def forward(self, x):
        x = F.reshape(x, self.out_shape)
        x = F.reshape(x, self.out_shape1)
        x = F.reshape(x, self.out_shape2)
        return x


class ElemwiseOpr(M.Module):
    def __init__(self, mode):
        super().__init__()
        self.data = np.ones((2, 3, 224, 224)).astype(np.float32)
        self.data1 = np.random.random((1, 3, 1, 1)).astype(np.float32)
        self.data2 = np.random.random((2, 3, 224, 224)).astype(np.float32) - 0.8
        self.mode = mode

    def forward(self, a):
        # add
        if self.mode == "add":
            x = a + mge.tensor(np.float32(10))
            y = a + mge.tensor(self.data1)
            z = x + y
        # sub
        elif self.mode == "sub":
            x = a - mge.tensor(np.float32(10))
            y = a - mge.tensor(self.data1)
            z = x - y
        # mul
        elif self.mode == "mul":
            x = a * mge.tensor(np.float32(10))
            y = mge.tensor(self.data1) * a
            z = x * y
        # div
        elif self.mode == "max":
            x = a + mge.tensor(self.data)
            y = a + mge.tensor(self.data2)
            z = F.maximum(x, y)
        elif self.mode == "min":
            x = a + mge.tensor(self.data)
            y = a + mge.tensor(self.data2)
            z = F.minimum(x, y)

        elif self.mode == "pow":
            z = a ** 2

        elif self.mode == "ceil":
            z = F.ceil(a)

        elif self.mode == "floor":
            z = F.floor(a)

        elif self.mode == "div":
            y = mge.tensor(self.data1) / a
            x = a / mge.tensor(np.float32(2))
            z = y / x
        # cycle_div
        elif self.mode == "cycle_div":
            z = a / mge.tensor(self.data1)
        # abs
        elif self.mode == "abs":
            z = F.abs(a)
        # exp
        elif self.mode == "exp":
            z = F.exp(a)
        # log
        elif self.mode == "log":
            z = F.log(a)
        elif self.mode == "fuse_add_relu":
            y = a + mge.tensor(self.data2)
            z = F.relu(y)
        elif self.mode == "fuse_mul_add3":
            y = a * mge.tensor(self.data1)
            z = y + mge.tensor(self.data2)
        elif self.mode == "fuse_add_sigmoid":
            y = a + mge.tensor(self.data2)
            z = F.sigmoid(y)
        else:
            raise NotImplementedError('no such elemwise mode "%s"' % self.mode)
        return z


class ReduceOpr(M.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.data = np.random.random((1, 3, 1000)).astype(np.float32)

    def forward(self, a):
        if self.mode == "sum":
            return F.sum(a, axis=2)
        elif self.mode == "mean":
            return F.mean(a, axis=2)
        else:
            return F.max(a, axis=2)


class ResizeOpr(M.Module):
    def __init__(self, mode="bilinear"):
        super().__init__()
        self.data = np.random.random((1, 2, 3, 4)).astype(np.float32)
        self.out_shape = [6, 8]
        self.out_shape2 = [3, 4]
        self.mode = mode

    def forward(self, x):
        x = F.vision.interpolate(x, size=self.out_shape, mode=self.mode)
        x = F.vision.interpolate(x, size=self.out_shape2, mode=self.mode)
        return x


class ActiveOpr(M.Module):
    str2fun = {
        "relu": F.relu,
        "tanh": F.tanh,
        "sigmoid": F.sigmoid,
        "leaky_relu": F.leaky_relu,
        "softmax": F.softmax,
        "relu6": F.relu6,
        "hswish": F.hswish,
        "hsigmoid": F.hsigmoid,
    }
    if mge.__version__ >= "1.5.0":
        str2fun["silu"] = F.silu

    def __init__(self, mode, fused=False):
        super().__init__()
        self.mode = mode
        self.fused = fused
        self.data = (np.random.random((1, 2, 3, 4)).astype(np.float32) - 0.5) * 8.0
        self.sigmoid = M.Sigmoid()
        self.act = ActiveOpr.str2fun[self.mode]

    def forward(self, x):
        if self.fused:
            return self.act(x + x)
        else:
            return self.act(x)


class BroadcastOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.array([1], dtype=np.float16)

    def forward(self, x):
        return F.broadcast_to(x, (3, 5))


class TypeCvtOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.array([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.int32)

    def forward(self, x):
        x = x + 1
        x = x.astype(np.float32)
        return x


class XORNet(M.Module):
    def __init__(self, converter="normal"):
        self.converter = converter
        self.mid_dim = 14
        self.num_class = 2
        super().__init__()
        self.fc0 = M.Linear(self.num_class, self.mid_dim, bias=True)
        self.bn0 = M.BatchNorm1d(self.mid_dim)
        self.fc1 = M.Linear(self.mid_dim, self.mid_dim, bias=True)
        self.bn1 = M.BatchNorm1d(self.mid_dim)
        self.fc2 = M.Linear(self.mid_dim, self.num_class, bias=True)
        self.data = np.arange(24).reshape(12, 2).astype(np.float32)

    def forward(self, x):
        x = self.fc0(x)
        x = self.bn0(x)
        x = F.softmax(x) if self.converter == "tflite" else F.tanh(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.softmax(x) if self.converter == "tflite" else F.tanh(x)
        x = self.fc2(x)
        return x


class XORNet_LeakyRelu(M.Module):
    def __init__(self):
        self.mid_dim = 14
        self.num_class = 2
        super().__init__()
        self.fc0 = M.Linear(self.num_class, self.mid_dim, bias=True)
        self.bn0 = M.BatchNorm1d(self.mid_dim)
        self.fc1 = M.Linear(self.mid_dim, self.mid_dim, bias=True)
        self.bn1 = M.BatchNorm1d(self.mid_dim)
        self.fc2 = M.Linear(self.mid_dim, self.num_class, bias=True)
        self.data = np.random.random((12, 2)).astype(np.float32)

    def forward(self, x):
        x = self.fc0(x)
        x = self.bn0(x)
        x = F.leaky_relu(x)
        x = F.leaky_relu(x)
        x = F.tanh(x)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.bn1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        return x


class RepeatOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((2, 3, 4)).astype("float32")

    def forward(self, x):
        x = F.repeat(x, 2, axis=1)
        return x


class FlattenOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((1, 2, 3, 4)).astype(np.float32)

    def forward(self, inps):
        return F.flatten(inps)


class DropoutOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((1, 2, 3, 4)).astype(np.float32)
        self.drop_out = M.Dropout()

    def forward(self, inps):
        return self.drop_out(inps)


class AdaptiveAvgPool2dOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((2, 512, 64, 64)).astype(np.float32)
        self.gap = M.AdaptiveAvgPool2d((2, 2))

    def forward(self, inps):
        return self.gap(inps)
