# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
        inference.dump(fpath + ".mge", optimize_for_inference=optimize_for_inference)
        return mge_result.numpy()


class ConvOpr(M.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.data = np.random.random((1, 3, 224, 224)).astype(np.float32)
        self.conv = M.Conv2d(3, 30, 3, stride=(2, 2), padding=(2, 2))
        self.group_conv = M.Conv2d(3, 30, 3, stride=(2, 2), padding=(2, 2), groups=3)

    def forward(self, x):
        if self.mode == 0:
            return self.conv(x)
        if self.mode == 1:
            return self.group_conv(x)


class LinearOpr(M.Module):
    def __init__(self):
        super().__init__()
        self.data = np.random.random((10, 100)).astype(np.float32)
        self.linear = M.Linear(100, 200, bias=False)
        self.linear_bias = M.Linear(200, 200, bias=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear_bias(x)
        return x


class PoolOpr(M.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.data = np.random.random((30, 3, 224, 224)).astype(np.float32)
        self.maxpool = M.pooling.MaxPool2d(kernel_size=3, stride=2, padding=2)
        self.avgpool = M.pooling.AvgPool2d(kernel_size=3, stride=2, padding=2)

    def forward(self, x):
        if self.mode == "max":
            return self.maxpool(x)
        if self.mode == "avg":
            return self.avgpool(x)


class BnOpr(M.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.data1 = np.random.random((1, 32, 32)).astype(np.float32)
        self.data2 = np.random.random((20, 3, 24, 24)).astype(np.float32)
        self.bn1d = M.BatchNorm1d(32)
        self.bn2d = M.BatchNorm2d(3)

    def forward(self, x):
        if self.mode == "bn1d":
            return self.bn1d(x)
        if self.mode == "bn2d":
            return self.bn2d(x)


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
            x = x[1, :, :, 3]
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
            return F.remove_axis(a, 0)
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
        self.data = np.ones((20, 3, 224, 224)).astype(np.float32)
        self.data1 = np.random.random((1, 3, 1, 1)).astype(np.float32)
        self.mode = mode

    def forward(self, a):
        # add
        if self.mode == "add":
            x = a + mge.tensor(np.float32(10))
            y = a + mge.tensor(self.data1)
            z = x + y
            return z
        # sub
        if self.mode == "sub":
            x = a - mge.tensor(np.float32(10))
            y = a - mge.tensor(self.data1)
            z = x - y
            return z
        # mul
        if self.mode == "mul":
            x = a * mge.tensor(np.float32(10))
            y = mge.tensor(self.data1) * a
            z = x * y
            return z
        # div
        if self.mode == "div":
            y = mge.tensor(self.data1) / a
            x = a / mge.tensor(np.float32(2))
            z = y / x
            return z
        # cycle_div
        if self.mode == "cycle_div":
            return a / mge.tensor(self.data1)
        # abs
        if self.mode == "abs":
            return F.abs(a)
        # exp
        if self.mode == "exp":
            return F.exp(a)
        # log
        if self.mode == "log":
            return F.log(a)


class ActiveOpr(M.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.data = np.random.random((20, 3, 224, 224)).astype(np.float32)

    def forward(self, x):
        if self.mode == "relu":
            return F.relu(x)
        if self.mode == "tanh":
            return F.tanh(x)
        if self.mode == "sigmoid":
            return F.sigmoid(x)


class XORNet(M.Module):
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
        x = F.tanh(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x
