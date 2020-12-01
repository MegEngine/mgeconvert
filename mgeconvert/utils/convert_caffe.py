# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse

from mgeconvert.caffe_converter import convert_to_caffe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Input megengine dump model file"
    )
    parser.add_argument(
        "-c", "--prototxt", required=True, type=str, help="Output caffe .prototxt file",
    )
    parser.add_argument(
        "-b",
        "--caffemodel",
        required=True,
        type=str,
        help="Output caffe .caffemodel file",
    )

    args = parser.parse_args()
    convert_to_caffe(args.input, prototxt=args.prototxt, caffemodel=args.caffemodel)


if __name__ == "__main__":
    main()
