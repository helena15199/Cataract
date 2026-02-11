# Cataract Phases Recognition

The goal here is to train a model for cataract phases classification.

First install everything:
```bash
uv sync
```

Then, prepare the dataset. Once you have your dataset you can run

```bash
uv run python train.py
```

Look at the `configs/config.yaml` to change the hyperparameters