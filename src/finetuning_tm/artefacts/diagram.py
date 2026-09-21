"""
Visualise road-extraction results: RGB | Ground Truth | one column per model,
one row per tile. Metrics under each prediction are the MEAN over the seed
parquet files in benchmarks/<model>/tiles/, looked up by tile_id.

You specify tiles + models; all file paths are derived from the tile_id (the
name is identical across imagery, masks, and predictions). No per-cell paths.

    python diagram.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile


# --------------------------------------------------------------------------- #
# style (thesis house style; falls back to sane defaults if plot_style absent)
# --------------------------------------------------------------------------- #
try:
    from plot_style import set_thesis_style
    set_thesis_style()
except Exception:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "figure.constrained_layout.use": True,
        "axes.spines.top": False, "axes.spines.right": False,
    })

MASK_CMAP = "gray"        # binary road masks; shared vmin/vmax below so panels are comparable
MASK_VMIN, MASK_VMAX = 0.0, 1.0


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_tiff(path):
    arr = np.asarray(tifffile.imread(str(path)))
    if arr.ndim == 3:
        # (C,H,W) if the first axis is the smallest -> move channels to last
        if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = np.moveaxis(arr, 0, -1)      # (C,H,W) -> (H,W,C)
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
    return arr


def stretch_rgb(arr: np.ndarray) -> np.ndarray:
    """2-98th percentile stretch to [0,1] for display of the RGB tile."""
    arr = arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    lo, hi = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    if hi <= lo:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
    return np.zeros_like(arr) if hi <= lo else np.clip((arr - lo) / (hi - lo), 0, 1)


# --------------------------------------------------------------------------- #
# model spec + seed-averaged metrics
# --------------------------------------------------------------------------- #
@dataclass
class Model:
    display_name: str
    pred_dir: Union[str, Path]
    tiles_dir: Union[str, Path]     # benchmarks/<model>/tiles  (cldice, apls)
    chips_dir: Union[str, Path]     # benchmarks/<model>/chips  (iou, f1 via counts)
    pred_suffix: str = "_pred.tif"


# pixel metrics live per-chip; pool them to the tile from counts (micro / pixel-weighted)
_CHIP_MICRO = {"iou", "f1", "precision", "recall", "accuracy"}


def _tile_micro(chips: pd.DataFrame, metric: str, tile_col="tile_id") -> pd.Series:
    g = chips.groupby(tile_col)[["tp", "fp", "fn", "tn"]].sum()
    tp, fp, fn, tn, eps = g.tp, g.fp, g.fn, g.tn, 1e-9
    return {
        "iou":       tp / (tp + fp + fn + eps),
        "f1":        2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall":    tp / (tp + fn + eps),
        "accuracy":  (tp + tn) / (tp + fp + fn + tn + eps),
    }[metric]


def mean_tile_metrics(tiles_dir, chips_dir, stat_columns, tile_col="tile_id"):
    """Per-tile mean over seeds. clDice/APLS from tiles/; IoU/F1 pooled from chips/.
    Returns (means_df indexed by tile_id, {'tiles': n, 'chips': n}, missing)."""
    tile_stats = [c for c in stat_columns if c not in _CHIP_MICRO]     # cldice, apls
    chip_stats = [c for c in stat_columns if c in _CHIP_MICRO]         # iou, f1, ...
    out, n, missing = None, {}, []

    if tile_stats:
        tf = sorted(Path(tiles_dir).glob("*.parquet"))
        if not tf: raise FileNotFoundError(f"no .parquet under {tiles_dir}")
        tdf = pd.concat([pd.read_parquet(f) for f in tf], ignore_index=True)
        present = [c for c in tile_stats if c in tdf.columns]
        missing += [c for c in tile_stats if c not in tdf.columns]
        out = tdf.groupby(tile_col)[present].mean()
        n["tiles"] = len(tf)

    if chip_stats:
        cf = sorted(Path(chips_dir).glob("*.parquet"))
        if not cf: raise FileNotFoundError(f"no .parquet under {chips_dir}")
        per_seed = []
        for f in cf:
            cdf = pd.read_parquet(f)
            st = pd.DataFrame(index=pd.Index(cdf[tile_col].unique(), name=tile_col))
            for m in chip_stats:
                st[m] = _tile_micro(cdf, m, tile_col)
            per_seed.append(st)
        chip_means = pd.concat(per_seed).groupby(level=0).mean()   # mean over seeds
        out = chip_means if out is None else out.join(chip_means, how="outer")
        n["chips"] = len(cf)

    return out, n, missing


def _rgb_path(root, tile, sub):  return Path(root) / sub / f"{tile}.tif"
def _gt_path(root, tile, sub):   return Path(root) / sub / f"{tile}.tif"
def _pred_path(m: "Model", tile):  return Path(m.pred_dir) / f"{tile}{m.pred_suffix}"


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
def load_strata(dataset_root, split="test", path_col="image_path",
                biome_col="biome", urban_col="urbanisation_classification"):
    """tile_id -> (biome, urbanisation) from splits/<split>.csv."""
    csv = Path(dataset_root) / "splits" / f"{split}.csv"
    df = pd.read_csv(csv)
    if path_col not in df.columns:
        raise KeyError(f"{path_col!r} not in {csv}; have {list(df.columns)}")
    key   = df[path_col].map(lambda p: Path(str(p)).stem)
    biome = df[biome_col] if biome_col in df.columns else ""
    urban = df[urban_col] if urban_col in df.columns else ""
    return dict(zip(key, zip(biome, urban)))


def _row_label(biome, urban):
    b = str(biome).replace("_", " ").title()                      # succulent_karoo -> Succulent Karoo
    u = {"periurban": "Peri-Urban", "peri_urban": "Peri-Urban"}.get(
        str(urban).strip().lower().replace(" ", "_"), str(urban))
    return f"{b}\n{u}"

def plot_results_grid(
    tiles: list[str],
    models: list[Model],
    dataset_root: Union[str, Path],
    stat_columns: list[str],
    imagery_subdir: str = "test/imagery",
    mask_subdir: str = "test/masks_raster",
    tile_col: str = "tile_id",
    figsize_per_cell: tuple[float, float] = (2.4, 2.4),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    split: str = "test",
):
    """Grid: [RGB | Ground Truth | model_1 | model_2 | ...], one row per tile.

    Prediction titles show the seed-mean of `stat_columns` for that tile.
    """
    metrics = {}
    strata = load_strata(dataset_root, split)
    for m in models:
        means, n, missing = mean_tile_metrics(m.tiles_dir, m.chips_dir, stat_columns, tile_col)
        metrics[m.display_name] = means
        if missing: print(f"[{m.display_name}] not found, skipped: {missing}")
        print(f"[{m.display_name}] averaged {n} seed file(s)")

    n_rows, n_cols = len(tiles), 2 + len(models)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
    )

    for r, tile in enumerate(tiles):
        top = (r == 0)

        ax = axes[r][0]
        ax.imshow(stretch_rgb(load_tiff(_rgb_path(dataset_root, tile, imagery_subdir))))
        if top: ax.set_title("S2 RGB")
        biome, urban = strata.get(tile, ("", ""))
        ax.set_ylabel(_row_label(biome, urban), rotation=0, ha="right",
                      va="center", labelpad=14, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

        ax = axes[r][1]
        ax.imshow(load_tiff(_gt_path(dataset_root, tile, mask_subdir)),
                  cmap=MASK_CMAP, vmin=MASK_VMIN, vmax=MASK_VMAX)
        if top: ax.set_title("Ground Truth")
        ax.set_axis_off()

        for c, m in enumerate(models):
            ax = axes[r][2 + c]
            ax.imshow(load_tiff(_pred_path(m, tile)),
                      cmap=MASK_CMAP, vmin=MASK_VMIN, vmax=MASK_VMAX)
            if top: ax.set_title(m.display_name)
            means = metrics[m.display_name]
            if tile in means.index:
                txt = "  ".join(f"{k} {means.loc[tile, k]:.3f}"
                                for k in stat_columns if k in means.columns)
            else:
                txt = "no metrics"
            ax.set_xlabel(txt, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=dpi)     # PNG >=300 dpi; no bbox_inches (keeps physical size)
        print(f"saved -> {save_path}")
    return fig


# --------------------------------------------------------------------------- #
# CONFIG — edit these, then run
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ROOT   = "/Users/catherineli/Desktop/InstaRoad/ROSADataset"
    BENCH  = "/Users/catherineli/Desktop/InstaRoad/InstaRoadPrototype/benchmarks"
    PREDS  = "/Users/catherineli/Desktop/InstaRoad/images/terratorch_predict"

    STAT_COLUMNS = ["iou", "apls"]        # tile-level metrics in benchmarks/*/tiles/

    def tiles(name): return f"{BENCH}/{name}/tiles"
    def chips(name): return f"{BENCH}/{name}/chips"

    MODELS = [
        Model("U-Net", pred_dir=f"{PREDS}/unet_baseline",            
              tiles_dir=tiles("unet_baseline"), chips_dir=chips("unet_baseline")),
        Model("D-LinkNet",  pred_dir=f"{PREDS}/dlinknet_baseline",
               tiles_dir=tiles("dlinknet_baseline"), chips_dir=chips("dlinknet_baseline")),
        Model("TM + FCN", pred_dir=f"{PREDS}/fcn",
              tiles_dir=tiles("tm_fcn"), chips_dir=chips("tm_fcn")),



    ]

    TILES = [
        "Forests_-28p55_29p08_Rural_r0_c2",
        "IndianOceanCoastalBelt_-30p81_30p34_PeriUrban_r4_c0",
        "CapeTown_Fynbos_-33p96_18p61_Urban_r0_c4"
    ]

    plot_results_grid(
        tiles=TILES, models=MODELS, dataset_root=ROOT,
        stat_columns=STAT_COLUMNS,
        save_path="/Users/catherineli/Desktop/InstaRoad/InstaRoadPrototype/src/finetuning_tm/results_grid.png",
    )