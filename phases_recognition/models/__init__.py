from omegaconf import DictConfig
import torch
from torch import nn
import torchvision.models as tv_models

from .cataract_predictor import CataractPredictor
from .mstcn import MSTCNPlusPlus, instantiate_mstcn

RESNET_VARIANTS = {
    "resnet18":  (tv_models.resnet18,  tv_models.ResNet18_Weights.DEFAULT),
    "resnet34":  (tv_models.resnet34,  tv_models.ResNet34_Weights.DEFAULT),
    "resnet50":  (tv_models.resnet50,  tv_models.ResNet50_Weights.DEFAULT),
    "resnet101": (tv_models.resnet101, tv_models.ResNet101_Weights.DEFAULT),
}

PREDICTORS = {
    "simple": CataractPredictor,
}


def build_pretrained_backbone(backbone_name: str, freeze: bool = False) -> tuple[nn.Module, int]:
    """
    Charge un ResNet préentraîné, retire sa tête FC,
    et retourne (backbone, feature_dim).
    """
    if backbone_name not in RESNET_VARIANTS:
        raise ValueError(f"Backbone '{backbone_name}' inconnu. Choix : {list(RESNET_VARIANTS)}")

    model_fn, weights = RESNET_VARIANTS[backbone_name]
    resnet = model_fn(weights=weights)

    feature_dim = resnet.fc.in_features  # 512 pour resnet18/34, 2048 pour resnet50/101

    # On retire la tête de classification
    backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, feature_dim


def instantiate_model(model_config: DictConfig | dict) -> nn.Module:
    name = model_config["name"]

    if name == "mstcn":
        return instantiate_mstcn(model_config)

    predictor_cls = PREDICTORS[name]
    num_classes = model_config.get("num_classes", 17)
    backbone_name = model_config["backbone"]["name"]  # ex: "resnet50"
    freeze = model_config["backbone"].get("freeze", False)

    backbone, final_size = build_pretrained_backbone(backbone_name, freeze=freeze)

    return predictor_cls(backbone=backbone, final_size=final_size, num_classes=num_classes)