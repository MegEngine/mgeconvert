# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse

from mgeconvert.cambricon_converter import convert_to_cambricon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="megengine dumped model file"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="converted Cambricon model file"
    )
    parser.add_argument(
        "-b", "--batch-size", default=4, type=int, help="best practice: 4"
    )
    parser.add_argument("-c", "--core-number", default=1, type=int, help="c <= 16")
    parser.add_argument(
        "-t", "--data-type", default="float32", type=str, help="float32, float16"
    )
    parser.add_argument("--use-nhwc", action="store_true", help="default nchw")

    args = parser.parse_args()

    convert_to_cambricon(
        args.input,
        args.output,
        args.batch_size,
        args.core_number,
        args.data_type,
        args.use_nhwc,
    )


if __name__ == "__main__":
    main()
