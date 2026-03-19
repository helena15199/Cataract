"""Visualizer for cataract phase classification."""

import pathlib

import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

matplotlib.set_loglevel("warning")


class ClassificationVisualizer:
    def __init__(
        self,
        img_dir: str,
        row: int = 4,
        col: int = 8,
        num_batch_to_take: int = 2,
        num_classes: int = 17,
        class_names: list[str] | None = None,
    ):
        self.img_dir = pathlib.Path(img_dir)
        self.row = row
        self.col = col
        self.num_batch_to_take = num_batch_to_take
        self.num_classes = num_classes
        # Optionnel : noms lisibles des phases ex ["Incision", "Capsulorhexis", ...]
        self.class_names = class_names

    def _label_name(self, idx: int) -> str:
        if self.class_names is not None and idx < len(self.class_names):
            return self.class_names[idx]
        return str(idx)

    @torch.no_grad()
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        paths: list[str],
        tag: str,
        batch_idx: int,
        global_step: int,
        epoch: int,
    ):
        if batch_idx >= self.num_batch_to_take:
            return

        probs = F.softmax(logits.detach().cpu(), dim=1)  # (N, 17)

        cmap = plt.cm.cividis
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])

        fig, axes = plt.subplots(self.row, self.col, figsize=(self.col * 2.5, self.row * 2.5))
        axes = axes.flatten()

        max_probs = []

        for ax_idx, ax in enumerate(axes):
            ax.axis("off")
            if ax_idx >= len(paths):
                continue

            img = cv2.imread(paths[ax_idx])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            prob = probs[ax_idx]
            pred = torch.argmax(prob).item()
            max_prob = prob[pred].item()
            gt = labels[ax_idx].item()

            max_probs.append(max_prob)

            correct = pred == gt
            title_color = "green" if correct else "red"
            edge_color = cmap(max_prob)

            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"GT: {self._label_name(gt)}\nPred: {self._label_name(pred)} ({max_prob:.2f})",
                color=title_color,
                fontsize=7,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(4)

        if max_probs:
            sm.set_clim(vmin=min(max_probs), vmax=max(max_probs))

        fig.colorbar(
            sm,
            ax=axes,
            orientation="horizontal",
            fraction=0.03,
            pad=0.05,
            label="Prediction confidence",
        )
        fig.suptitle(f"{tag} | {epoch=} | {global_step=} | {batch_idx=}", fontsize=10)
        fig.tight_layout()

        outdir = self.img_dir / tag
        outdir.mkdir(exist_ok=True, parents=True)
        fig.savefig(
            outdir / f"epoch={epoch:04d}_step={global_step:06d}_batch={batch_idx:02d}.png",
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(fig)


def instantiate_visualizer(
    img_dir: str,
    row: int = 4,
    col: int = 8,
    num_batch_to_take: int = 2,
    num_classes: int = 17,
    class_names: list[str] | None = None,
) -> ClassificationVisualizer:
    return ClassificationVisualizer(
        img_dir=img_dir,
        row=row,
        col=col,
        num_batch_to_take=num_batch_to_take,
        num_classes=num_classes,
        class_names=class_names,
    )