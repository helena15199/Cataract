"""End-to-end model: ResNet50 backbone + MS-TCN++.

Backbone and TCN are trained jointly. The backbone learns features optimised
for temporal phase discrimination instead of per-frame classification.

forward(frames):
    frames  : (T, 3, H, W)  — one window of consecutive frames
    returns : list of (1, num_classes, T) logits, one per MS-TCN stage
"""

import torch
import torch.nn as nn

import torchvision.models as tv_models

from .mstcn import MSTCNPlusPlus

_RESNET_VARIANTS = {
    "resnet18":  (tv_models.resnet18,  tv_models.ResNet18_Weights.DEFAULT),
    "resnet34":  (tv_models.resnet34,  tv_models.ResNet34_Weights.DEFAULT),
    "resnet50":  (tv_models.resnet50,  tv_models.ResNet50_Weights.DEFAULT),
    "resnet101": (tv_models.resnet101, tv_models.ResNet101_Weights.DEFAULT),
}


def _build_backbone(name: str) -> nn.Module:
    model_fn, weights = _RESNET_VARIANTS[name]
    resnet = model_fn(weights=weights)
    return nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())


class E2ETemporalModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tcn: MSTCNPlusPlus,
    ):
        super().__init__()
        self.backbone = backbone  # ResNet50 without FC head → (T, 2048)
        self.tcn      = tcn       # MS-TCN++ → list of (1, C, T)

    def forward(self, frames: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            frames: (T, 3, H, W) — backbone processes all T frames as a flat batch
        Returns:
            list of (1, num_classes, T) logits, one per stage
        """
        features = self.backbone(frames)              # (T, 2048)
        features = features.unsqueeze(0)              # (1, T, 2048)
        return self.tcn(features)                     # list of (1, C, T)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def instantiate_e2e_model(model_config) -> E2ETemporalModel:
    backbone_name = model_config.get("backbone_name", "resnet50")
    backbone = _build_backbone(backbone_name)

    # Charge le checkpoint cataracte si fourni — bien mieux que partir d'ImageNet seul
    cataract_ckpt = model_config.get("cataract_backbone_ckpt", None)
    if cataract_ckpt:
        state = torch.load(cataract_ckpt, map_location="cpu", weights_only=False)
        # Le checkpoint vient de CataractPredictor : backbone.X → on extrait juste backbone
        backbone_state = {
            k.removeprefix("backbone."): v
            for k, v in state["model_state_dict"].items()
            if k.startswith("backbone.")
        }
        backbone.load_state_dict(backbone_state)
        print(f"Loaded cataract backbone from {cataract_ckpt}")
    else:
        print("No cataract_backbone_ckpt provided — using ImageNet weights only.")

    tcn = MSTCNPlusPlus(
        num_stages  = model_config.get("num_stages",  4),
        num_layers  = model_config.get("num_layers",  10),
        num_f_maps  = model_config.get("num_f_maps",  64),
        input_dim   = model_config.get("input_dim",   2048),
        num_classes = model_config.get("num_classes", 13),
        dropout     = model_config.get("dropout",     0.5),
    )
    return E2ETemporalModel(backbone=backbone, tcn=tcn)
