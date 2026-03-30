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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.backbone(image)
        complete = self.fc_complete_number(x)
        return complete