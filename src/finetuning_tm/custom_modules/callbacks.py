"""Predict-time comparison panels for terratorch's CLI.

Creates png imges from the dataset split that contains:
1. the satellite image with RGB channel + ground truth road labels, 
2. the ground truth road label mask 
3. the predicted road mask 
4. Performance metrics (IoU, F1, accuracy, precision, recall) of the tile

Active only during ``trainer.predict``. Panels land in
``<output_dir>/<split>/`` where ``<split>`` is read from the datamodule's
``predict_split``, so repointing the datamodule moves the output with it.

Wire it under ``trainer.callbacks`` in the predict config:

```yaml
    - class_path: custom_modules.callbacks.PredictionWriter
      init_args:
        output_dir: /kaggle/working/predictions/panels
        rgb_key: S2L2A
        rgb_indices: [0, 1, 2]
        positive_class: 1
        filename_stat: iou
```
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # HPC compute nodes have no display
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.callbacks import Callback

_STAT_KEYS = ("iou", "f1", "accuracy", "precision", "recall")


def _stretch_any(img_chw: np.ndarray, percentile_range=(2, 98)) -> np.ndarray:
    img = np.nan_to_num(img_chw, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        band = img[i]
        lo, hi = np.percentile(band, list(percentile_range))
        if hi > lo:
            out[i] = np.clip((np.clip(band, lo, hi) - lo) / (hi - lo), 0, 1)
    return (np.transpose(out, (1, 2, 0)) * 255).astype(np.uint8)

def _overlay(rgb, mask, color=(255, 0, 0), alpha=0.45):
    """Blend a binary mask over an (H, W, 3) uint8 RGB image."""
    out = rgb.astype(np.float32).copy()
    sel = mask > 0
    for c in range(3):
        out[..., c][sel] = (1.0 - alpha) * out[..., c][sel] + alpha * color[c]
    return out.astype(np.uint8)

class PredictionWriter(Callback):
    """Per-tile panels + per-tile metrics. Predict loop only."""

    def __init__(
        self,
        output_dir: str = "prediction_images",
        threshold: float = 0.5,
        rgb_key: str = "S2L2A",
        rgb_indices: tuple[int, int, int] = (0, 1, 2),
        positive_class: int = 1,
        filename_stat: str = "iou",
        overlay_alpha: float = 0.45,
        overlay_color: tuple[int, int, int] = (255, 0, 0),
        dpi: int = 150,
    ):
        super().__init__()
        if filename_stat not in _STAT_KEYS:
            raise ValueError(f"filename_stat must be one of {_STAT_KEYS}, got {filename_stat!r}")
        self.output_dir = output_dir
        self.threshold = threshold
        self.rgb_key = rgb_key
        self.rgb_indices = rgb_indices
        self.positive_class = positive_class
        self.filename_stat = filename_stat
        self.overlay_alpha = overlay_alpha
        self.overlay_color = overlay_color
        self.dpi = dpi
        self._split_dir = None
        self._reset()

    def _reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0
        self._n_saved = 0

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _unpack_predict(outputs):
        """predict_step returns (select_classes(y_hat), file_names).

        select_classes yields (tensor, suffix) for a str `output_on_inference`,
        or a list of those tuples for a list.
        """
        preds, filenames = outputs
        if isinstance(preds, list):          # multiple selectors -> first one
            preds = preds[0]
        tensor, _suffix = preds
        return tensor, filenames

    def _probs_from(self, raw):
        if raw.dim() == 4:                              # logits / probabilities (B,C,H,W)
            return raw[:, self.positive_class]
        return (raw == self.positive_class).float()      # argmax labels (B,H,W)

    @staticmethod
    def _confusion(pred, gt):
        return (float(np.sum(pred & gt)), float(np.sum(pred & ~gt)),
                float(np.sum(~pred & gt)), float(np.sum(~pred & ~gt)))

    @staticmethod
    def _metrics(tp, fp, fn, tn):
        eps = 1e-6
        return {
            "iou": tp / (tp + fp + fn + eps),
            "f1": 2 * tp / (2 * tp + fp + fn + eps),
            "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
        }

    # -- hooks -----------------------------------------------------------
    def on_predict_start(self, trainer, pl_module):
        self._reset()
        dm = getattr(trainer, "datamodule", None)
        split = getattr(dm, "predict_split", None) or "predict"
        self._split_dir = os.path.join(self.output_dir, split)
        os.makedirs(self._split_dir, exist_ok=True)
        print(f"PredictionWriter -> {self._split_dir}")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.is_global_zero or outputs is None:
            return
        raw, _suffix_names = self._unpack_predict(outputs)
        probs = self._probs_from(raw).detach().cpu().numpy()

        masks = batch["mask"].detach().cpu().numpy() if "mask" in batch else None
        img = batch["image"]
        rgb_stack = (img[self.rgb_key] if isinstance(img, dict) else img).detach().cpu().numpy()  # (B, C, H, W)
        filenames = batch.get("filename", [""] * len(probs))

        for i in range(len(probs)):
            rgb = _stretch_any(rgb_stack[i][list(self.rgb_indices)])
            pred_mask = probs[i] > self.threshold
            true_mask = (masks[i] > 0) if masks is not None else None
            self._save_one(rgb, pred_mask, true_mask, filenames[i])

    def on_predict_epoch_end(self, trainer, pl_module):
        # NB: no pl_module.log_dict here — Lightning forbids logging in predict.
        if not trainer.is_global_zero:
            return
        total = self.tp + self.fp + self.fn + self.tn
        if total == 0:
            print(f"PredictionWriter: wrote {self._n_saved} panels (no masks, no metrics).")
            return
        m = self._metrics(self.tp, self.fp, self.fn, self.tn)
        print(f"\nDataset-wide metrics over {self._n_saved} tiles (scored once per pixel):")
        for k, v in m.items():
            print(f"  - {k}: {round(float(v), 4)}")

    # -- writing ---------------------------------------------------------
    def _save_one(self, rgb, pred_mask, true_mask, filename):
        stem = Path(filename).stem or f"tile{self._n_saved:05d}"

        if true_mask is not None:
            tp, fp, fn, tn = self._confusion(pred_mask, true_mask)
            self.tp += tp; self.fp += fp; self.fn += fn; self.tn += tn
            m = self._metrics(tp, fp, fn, tn)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(_overlay(rgb, true_mask, self.overlay_color, self.overlay_alpha))
            axes[0].set_title("Image + Ground Truth")
            axes[1].imshow(true_mask, cmap="gray"); axes[1].set_title("Ground Truth")
            axes[2].imshow(pred_mask, cmap="gray"); axes[2].set_title("Prediction")
            fig.suptitle(
                f"{stem}\n"
                f"IoU {m['iou']:.3f}   F1 {m['f1']:.3f}   Acc {m['accuracy']:.3f}   "
                f"Prec {m['precision']:.3f}   Rec {m['recall']:.3f}",
                fontsize=10,
            )
            out_name = f"{stem}_{self.filename_stat}{m[self.filename_stat]:.4f}.png"
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(rgb); axes[0].set_title("Image (RGB)")
            axes[1].imshow(pred_mask, cmap="gray"); axes[1].set_title("Prediction")
            fig.suptitle(stem, fontsize=10)
            out_name = f"{stem}.png"

        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(self._split_dir, out_name), bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)
        self._n_saved += 1