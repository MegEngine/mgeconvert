import csv
import os
import sys

from basecls_model import (
    effnet_models,
    hrnet_models,
    mb_models,
    regnet_models,
    resnet_models,
    snet_models,
    vgg_models,
    vit_models,
)

if __name__ == "__main__":
    with open("out.csv", "w") as f:
        f.truncate()
        writer = csv.writer(f)
        writer.writerow(["model", "framework", "absolute error", "relative error"])

    for model in (
        resnet_models
        + regnet_models
        + vgg_models
        + effnet_models
        + vit_models
        + hrnet_models
        + snet_models
        + mb_models
    ):
        for framework in ["caffe", "tflite", "onnx"]:
            out = os.system(
                "python3 test_converter.py {} {} out.csv".format(model, framework)
            )
            if out != 0:
                sys.exit()
