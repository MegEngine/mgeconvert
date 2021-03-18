# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse

from mgeconvert.tflite_converter import convert_to_tflite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="megengine dumped model file"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="converted TFLite model file"
    )
    parser.add_argument(
        "--graph-name",
        default="graph0",
        type=str,
        help="default subgraph name in TFLite model",
    )
    parser.add_argument(
        "-b", "--batch-size", default=1, type=int, help="default value: 1"
    )
    parser.add_argument(
        "--mtk", action="store_true", help="If target flatform is MTK(P70, P80)"
    )

    args = parser.parse_args()

    convert_to_tflite(
        mge_fpath=args.input,
        output=args.output,
        graph_name=args.graph_name,
        batch_size=args.batch_size,
        mtk=args.mtk,
    )


if __name__ == "__main__":
    main()
