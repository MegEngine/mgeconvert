# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# pylint: disable=import-error,no-name-in-module

from megengine.traced_module import trace_module


def get_traced_module(net, *x):
    traced_module = trace_module(net, *x)
    expect = traced_module(*x)
    return traced_module, expect
