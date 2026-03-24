import math
import torch


def get_linear_warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    n_warmup_steps: int,
):
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min((step + 1) / n_warmup_steps, 1.0)
    )
    return warmup


def get_linear_warmup_cosine_decay_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    n_warmup_steps: int,
    n_total_steps: int,
    min_lr_ratio: float = 0.0,
):
    def lr_lambda(step: int) -> float:
        if step < n_warmup_steps:
            return (step + 1) / n_warmup_steps
        progress = (step - n_warmup_steps) / max(n_total_steps - n_warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)