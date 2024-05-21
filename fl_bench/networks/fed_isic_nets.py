import torch
import timm

from fl_bench.utils.misc import seed_everything


model_dir = # Path to the directory where the models are stored
torch.hub.set_dir(model_dir)

NUM_CLASSES = 8

def resnet_50_pretrained_timm(seed):
    seed_everything(seed)
    bb = timm.create_model(
        'resnet50.ra_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES)
    return bb

resnet_50_supervised = resnet_50_pretrained_timm
resnet_50_pretrained = resnet_50_pretrained_timm


def nf_resnet50_pretrained(seed):
    seed_everything(seed)
    model = timm.create_model(
        'nf_resnet50.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    return model

nf_resnet50_supervised = nf_resnet50_pretrained


def seresnet_50_supervised(seed):
    seed_everything(seed)
    model = timm.create_model(
        'seresnet50.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    return model


def anfr_resnet50_pretrained(seed):
    model = timm.create_model(
        "nf_seresnet50",
        pretrained=True,
        pretrained_cfg_overlay=dict(file=''), # Path to the pretrained model
        num_classes=NUM_CLASSES
    )
    return model


def resnet50_gn_supervised(seed):
    seed_everything(seed)
    model = timm.create_model(
        'resnet50_gn.a1h_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    return model
