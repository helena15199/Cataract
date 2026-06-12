"""Ensemble evaluation: MSTCN DINOv2 + LSTM DINOv2.

Options tested:
  A) Soft voting      — average softmax probs
  B) Weighted voting  — per-class weight from val F1
  C) Oracle           — frame-by-frame best model (upper bound)

Usage (from repo root):
    python phases_recognition/ensemble_eval.py
"""

import pathlib
import sys
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from metrics.cataract_metrics import CataractMetrics
from models import instantiate_model

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

MSTCN_EXP  = "/home/helena/experiments_cataract/mstcn_dino_v1_date=2026_06_11_17_02_41"
LSTM_EXP   = "/home/helena/experiments_cataract/lstm_dino_v1_date=2026_06_11_17_40_27"
TEST_ROOT  = "/home/helena/UCL_video_cataract/features_dino/test/"

# Per-class val F1 from individual evaluations (used for weighted voting)
# Order matches class_names below
MSTCN_VAL_F1 = np.array([0.156, 0.634, 0.888, 0.587, 0.862, 0.920,
                          0.799, 0.327, 0.540, 0.801, 0.766])
LSTM_VAL_F1  = np.array([0.595, 0.607, 0.928, 0.527, 0.818, 0.874,
                          0.666, 0.271, 0.380, 0.692, 0.758])

CLASS_NAMES = [
    "Capsule_polishing", "Hydrodissection", "Incision",
    "Irrigation_and_aspiration", "Lens_implant_settingup",
    "Phacoemulsification", "Rhexis", "Tonifying_and_antibiotics",
    "Viscous_agent_injection", "Viscous_agent_removal", "Wound_hydration",
]

# ---------------------------------------------------------------------------

def load_model(exp_dir: str, device: torch.device):
    cfg   = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_model(cfg.model)
    state = torch.load(f"{exp_dir}/ckpt/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def get_probs(model, loader, device):
    """Returns dict {video_name: (probs (T,C), labels (T,))}"""
    results = {}
    for features, labels, video_name in loader:
        feat = features.unsqueeze(0).to(device)
        logits = model(feat)[-1].squeeze(0).T        # (T, C)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        results[video_name] = (probs, labels.numpy())
    return results


def compute_metrics(video_results, num_classes, class_names, tag):
    """video_results: list of (gt_seq, pred_seq, video_name)"""
    metrics_fn = CataractMetrics(num_classes=num_classes,
                                  class_names=class_names,
                                  others_classes=[])
    for gt, pred, _ in video_results:
        T = len(gt)
        dummy = torch.zeros(T, num_classes)
        dummy[range(T), pred] = 10.0
        metrics_fn.update(dummy, torch.tensor(gt))
    m = metrics_fn.compute()
    f1  = m.get("global/f1_macro", 0.0)
    acc = m.get("global/accuracy",  0.0)
    per_class = {c: m.get(f"per_class/f1/{c}", 0.0) for c in class_names}
    print(f"\n{'='*55}")
    print(f"  {tag}")
    print(f"{'='*55}")
    print(f"  F1 macro : {f1:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"\n  Per-class F1:")
    for c, v in per_class.items():
        print(f"    {c:<35} {v:.3f}")
    return f1, acc, per_class


# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    mstcn = load_model(MSTCN_EXP, device)
    lstm  = load_model(LSTM_EXP,  device)

    dataset = VideoFeatureDataset(root=TEST_ROOT)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         collate_fn=_collate_single_video)

    print(f"Running inference on {len(dataset)} test videos...")
    probs_mstcn = get_probs(mstcn, loader, device)
    probs_lstm  = get_probs(lstm,  loader, device)

    num_classes = len(CLASS_NAMES)

    # Per-class weights (normalised so they sum to 1 per class)
    total = MSTCN_VAL_F1 + LSTM_VAL_F1
    w_mstcn = MSTCN_VAL_F1 / np.maximum(total, 1e-6)   # (C,)
    w_lstm  = LSTM_VAL_F1  / np.maximum(total, 1e-6)   # (C,)

    results_soft     = []
    results_weighted = []
    results_oracle   = []
    results_mstcn    = []
    results_lstm     = []

    for vname in probs_mstcn:
        pm, labels = probs_mstcn[vname]   # (T, C)
        pl, _      = probs_lstm[vname]

        # Individual
        pred_m = pm.argmax(axis=1).tolist()
        pred_l = pl.argmax(axis=1).tolist()

        # A — soft voting
        pred_soft = ((pm + pl) / 2).argmax(axis=1).tolist()

        # B — weighted voting (per-class)
        # pm[:, c] * w_mstcn[c] + pl[:, c] * w_lstm[c]
        pm_w = pm * w_mstcn[None, :]   # (T, C)
        pl_w = pl * w_lstm [None, :]
        pred_weighted = (pm_w + pl_w).argmax(axis=1).tolist()

        # C — oracle: frame-by-frame pick the model that's right
        gt = labels.tolist()
        correct_m = (np.array(pred_m) == labels).astype(int)
        correct_l = (np.array(pred_l) == labels).astype(int)
        oracle = []
        for t in range(len(gt)):
            if correct_m[t]:
                oracle.append(pred_m[t])
            elif correct_l[t]:
                oracle.append(pred_l[t])
            else:
                oracle.append(pred_m[t])   # both wrong → arbitrary
        pred_oracle = oracle

        results_mstcn.append((gt, pred_m,        vname))
        results_lstm.append( (gt, pred_l,        vname))
        results_soft.append( (gt, pred_soft,     vname))
        results_weighted.append((gt, pred_weighted, vname))
        results_oracle.append(  (gt, pred_oracle,   vname))

    compute_metrics(results_mstcn,    num_classes, CLASS_NAMES, "MSTCN DINOv2 (individual)")
    compute_metrics(results_lstm,     num_classes, CLASS_NAMES, "LSTM DINOv2 (individual)")
    compute_metrics(results_soft,     num_classes, CLASS_NAMES, "A — Soft voting (avg probs)")
    compute_metrics(results_weighted, num_classes, CLASS_NAMES, "B — Weighted voting (per-class val F1)")
    compute_metrics(results_oracle,   num_classes, CLASS_NAMES, "C — Oracle (upper bound)")

    print("\n" + "="*55)
    print("Summary")
    print("="*55)


if __name__ == "__main__":
    main()
