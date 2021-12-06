from basecls.configs import (
    EffNetConfig,
    EffNetLiteConfig,
    HRNetConfig,
    MBConfig,
    RegNetConfig,
    ResNetConfig,
    SNetConfig,
    VGGConfig,
    ViTConfig,
)
from basecls.models import build_model

resnet_models = [
    # "resnet101",
    # "resnet101d",
    # "resnet152",
    # "resnet152d",
    "resnet18",
    # "resnet18d",
    # "resnet34",
    # "resnet34d",
    # "resnet50",
    # "resnet50d",
    # "resnext101_32x4d",
    # "resnext101_32x8d",
    # "resnext101_64x4d",
    # "resnext152_32x4d",
    # "resnext152_32x8d",
    # "resnext152_64x4d",
    # "resnext50_32x4d",
    # "se_resnet101",
    # "se_resnet152",
    # "se_resnet18",
    # "se_resnet34",
    # "se_resnet50",
    # "se_resnext101_32x4d",
    # "se_resnext101_32x8d",
    # "se_resnext101_64x4d",
    # "se_resnext152_32x4d",
    # "se_resnext152_32x8d",
    # "se_resnext152_64x4d",
    # "se_resnext50_32x4d",
    # "wide_resnet101_2",
    # "wide_resnet50_2",
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
    "vit_base_patch16_384",
    "vit_base_patch32_224",
    "vit_base_patch32_384",
    "vit_small_patch16_224",
    "vit_small_patch16_384",
    "vit_small_patch32_224",
    "vit_small_patch32_384",
    "vit_tiny_patch16_224",
    "vit_tiny_patch16_384",
]

hrnet_models = [
    "hrnet_w18",
    "hrnet_w18_small_v1",
    "hrnet_w18_small_v2",
    "hrnet_w30",
    "hrnet_w32",
    "hrnet_w40",
    "hrnet_w44",
    "hrnet_w48",
    "hrnet_w64",
]

snet_models = ["snetv2_x050", "snetv2_x100", "snetv2_x150", "snetv2_x200"]

mb_models = [
    "mbnetv1_x025",
    "mbnetv1_x075",
    "mbnetv1_x100",
    "mbnetv2_x035",
    "mbnetv2_x050",
    "mbnetv2_x075",
    "mbnetv2_x100",
    "mbnetv2_x140",
    "mbnetv3_large_x075",
    "mbnetv3_large_x100",
    "mbnetv3_small_x075",
    "mbnetv3_small_x100",
]


_cls_configs = {
    "resnet": ResNetConfig,
    "resnext": ResNetConfig,
    "regnet": RegNetConfig,
    "effnet": EffNetConfig,
    "vgg": VGGConfig,
    "vit": ViTConfig,
    "hr": HRNetConfig,
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
        resnet_models
    )
    return PublicClsModelIter(public_cls_models)
