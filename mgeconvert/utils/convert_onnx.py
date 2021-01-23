# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse

from mgeconvert.onnx_converter import convert_to_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Input megengine dump model file"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="Output onnx .onnx file"
    )
    parser.add_argument("--opset", default=8, type=int, help="Onnx opset version")
    parser.add_argument("--graph", default="graph", type=str, help="Onnx graph name")
    parser.add_argument(
        "--end_point",
        default=None,
        type=str,
        help="end_point is used to specify which part of the mge model should be converted",
    )
    args = parser.parse_args()
    outspec = None
    if args.end_point is not None:
        outspec = args.end_point.split(";")

    convert_to_onnx(
        args.input,
        args.output,
        graph_name=args.graph,
        opset=args.opset,
        outspec=outspec,
    )


if __name__ == "__main__":
    main()
