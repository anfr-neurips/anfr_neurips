import os
import timm
import torch

from fl_bench.utils.misc import seed_everything

model_dir = # Path to the directory where the models are stored
torch.hub.set_dir(model_dir)

NUM_CLASSES = 8


def resnet_50_supervised(seed):
    seed_everything(seed)
    bb = timm.create_model(
        'resnet50.ra_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return bb

resnet_50_pretrained = resnet_50_supervised

def seresnet_50_supervised(seed):
    # Image size: train = 224 x 224, test = 288 x 288
    seed_everything(seed)
    model = timm.create_model(
        'seresnet50.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return model


def nf_resnet50_supervised(seed):
    seed_everything(seed)
    model = timm.create_model(
        'nf_resnet50.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return model


def resnet50_gn_supervised(seed):
    seed_everything(seed)
    model = timm.create_model(
        'resnet50_gn.a1h_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return model


def anfr_resnet50_pretrained(seed):
    model = timm.create_model(
        "nf_seresnet50",
        pretrained=True,
        pretrained_cfg_overlay=dict(file=),# Path to the pretrained model)
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return model
