"""Visualizer for cataract phase classification."""

import pathlib

import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


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
        self.class_names = class_names
        self.writer = None  # injecté par le trainer après création du SummaryWriter

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
            ax.set_xticks([])
            ax.set_yticks([])
            if ax_idx >= len(paths):
                ax.axis("off")
                continue

            img = cv2.imread(paths[ax_idx])
            if img is None:
                ax.axis("off")
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
            ax.set_title(
                f"GT: {self._label_name(gt)}\nPred: {self._label_name(pred)} ({max_prob:.2f})",
                color=title_color,
                fontsize=7,
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(6)

        if max_probs:
            sm.set_clim(vmin=min(max_probs), vmax=max(max_probs))

        fig.suptitle(f"{tag} | {epoch=} | {global_step=} | {batch_idx=}", fontsize=10)
        fig.tight_layout(rect=[0, 0.05, 1, 1])

        cbar_ax = fig.add_axes([0.15, 0.01, 0.7, 0.02])
        fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="Prediction confidence")

        outdir = self.img_dir / tag
        outdir.mkdir(exist_ok=True, parents=True)
        fig.savefig(
            outdir / f"epoch={epoch:04d}_step={global_step:06d}_batch={batch_idx:02d}.png",
            dpi=100,
            bbox_inches="tight",
        )
        if self.writer is not None:
            self.writer.add_figure(f"{tag}/predictions_batch{batch_idx}", fig, global_step=epoch)
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


class TemporalVisualizer:
    """
    Génère une timeline GT vs Prédiction pour chaque vidéo de validation.
    Sauvegarde les figures sur disque et les logue dans TensorBoard.

    Appelé à chaque epoch de validation avec les séquences accumulées.
    """

    def __init__(self, img_dir: str, class_names: list[str]):
        self.img_dir = pathlib.Path(img_dir)
        self.class_names = class_names
        self.writer = None  # injecté par le trainer

        n = len(class_names)
        cmap = matplotlib.colormaps.get_cmap("tab20")
        self._colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    def _draw_timeline(
        self,
        gt_seq: list[int],
        pred_seq: list[int],
        video_name: str,
    ) -> plt.Figure:
        n_frames = len(gt_seq)
        fig, axes = plt.subplots(2, 1, figsize=(14, 2.0), squeeze=False)

        for row_idx, (seq, label) in enumerate([(gt_seq, "GT"), (pred_seq, "Pred")]):
            ax = axes[row_idx, 0]
            start = 0
            for i in range(1, n_frames + 1):
                if i == n_frames or seq[i] != seq[start]:
                    left  = start / n_frames
                    width = (i - start) / n_frames
                    ax.barh(0, width, left=left, color=self._colors[seq[start]],
                            height=0.8, align="center")
                    start = i
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([0])
            ax.set_yticklabels([label], fontsize=9)
            ax.set_xticks([])
            for spine in ["top", "right", "bottom"]:
                ax.spines[spine].set_visible(False)

        handles = [plt.Rectangle((0, 0), 1, 1, color=self._colors[i])
                   for i in range(len(self.class_names))]
        fig.legend(handles, self.class_names,
                   loc="lower center", ncol=min(len(self.class_names), 7),
                   fontsize=7, bbox_to_anchor=(0.5, -0.35), frameon=False)
        fig.suptitle(video_name, fontsize=9, x=0.01, ha="left")
        fig.tight_layout(rect=[0, 0.15, 1, 1])
        return fig

    @torch.no_grad()
    def log_epoch(
        self,
        video_sequences: list[tuple[list[int], list[int], str]],
        epoch: int,
    ):
        """
        Args:
            video_sequences: liste de (gt_seq, pred_seq, video_name) pour toutes les vidéos de val
            epoch: numéro d'epoch courant
        """
        out_dir = self.img_dir / f"epoch_{epoch:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for gt_seq, pred_seq, video_name in video_sequences:
            fig = self._draw_timeline(gt_seq, pred_seq, video_name)
            safe_name = video_name[:60].replace("/", "_")
            fig.savefig(out_dir / f"{safe_name}.png", dpi=80, bbox_inches="tight")
            if self.writer is not None:
                self.writer.add_figure(f"val/timeline/{safe_name}", fig, global_step=epoch)
            plt.close(fig)