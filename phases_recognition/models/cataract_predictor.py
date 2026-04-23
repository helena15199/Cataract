"""General strategy fo the cataract predictor."""

import torch
from torch import nn


class CataractPredictor(nn.Module):
    def __init__(self, backbone: nn.Module, final_size: int, num_classes: int = 17):
        super().__init__()
        self.num_classes = num_classes
        self.final_size = final_size

        self.backbone = backbone
        self.fc_complete_number = nn.Linear(self.final_size, num_classes)
        self.fc_binary = nn.Linear(self.final_size, 1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(image)
        return self.fc_complete_number(x), self.fc_binary(x).squeeze(1)