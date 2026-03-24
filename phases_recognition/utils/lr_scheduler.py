import torch


def get_linear_warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    n_warmup_steps: int,
):
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min((step + 1) / n_warmup_steps, 1.0)
    )
    return warmup