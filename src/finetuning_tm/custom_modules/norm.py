"""
    to adapt the dataset to the datamodule for terramind 
"""
import torch 
import numpy as np
from terratorch.models.backbones.terramind.model.terramind_register import PRETRAINED_BANDS
from terratorch.models.backbones.terramind.model.terramind_register import (
    v1_pretraining_mean, v1_pretraining_std, PRETRAINED_BANDS,
)

S2L2A_NON_RGB = {
    "RED_EDGE_1": "B5",
    "RED_EDGE_2": "B6",
    "RED_EDGE_3": "B7",
    "NIR_NARROW": "B8A",
    "NIR_BROAD":  "B8_NIR",
    "SWIR_1":     "B11",
    "SWIR_2":     "B12",
}

RGB_SOURCES = {
    "raw":      {"RED": "B4_R",          "GREEN": "B3_G",          "BLUE": "B2_B"},
    "enhanced": {"RED": "B4_R_enhanced", "GREEN": "B3_G_enhanced", "BLUE": "B2_B_enhanced"},
}
S1GRD_BANDS = {
    "VV": "VV_ascending",
    "VH": "VH_ascending",
}

# raster band index -> descriptive band name, per your dataset's numbering
BAND_INDEX_TO_NAME = {
    1: "B4_R", 
    2: "B3_G", 
    3: "B2_B",
    4:  "B8_NIR",
    5:  "B5",
    6:  "B6",
    7:  "B7",
    8:  "B8A",
    9:  "B11",
    10: "B12",
    11: "VV_ascending",
    12: "VH_ascending",
    21: "B4_R_enhanced",
    22: "B3_G_enhanced",
    23: "B2_B_enhanced",
}

######
# to adapt the normalisation stats of terramind 

# TerraMind's pretraining stats are fit on specific *unscaled* inputs:
#   - untok_sen2l2a@224: raw Sentinel-2 L2A DN, ~0-10000 (our data is DN/10000)
#   - untok_sen2rgb@224: natural-image RGB, 0-255 (our *_enhanced bands are a
#     [0,1] min-max stretch of that same display-like product)
#   - untok_sen1grd@224: dB backscatter (matches our data as-is)
TM_MODALITY_S2L2A = "untok_sen2l2a@224"
TM_MODALITY_S1GRD = "untok_sen1grd@224"
TM_MODALITY_IDENTITY = "identity"

TM_RESCALE = {
    TM_MODALITY_S2L2A: 10000.0,
    TM_MODALITY_S1GRD: 1.0,
    TM_MODALITY_IDENTITY: 1.0,
}

# our band name (BAND_INDEX_TO_NAME) -> (terramind modality key, band name in PRETRAINED_BANDS[modality])
TM_STATS_KEY = {
    "B4_R":         (TM_MODALITY_S2L2A, "RED"),
    "B3_G":         (TM_MODALITY_S2L2A, "GREEN"),
    "B2_B":         (TM_MODALITY_S2L2A, "BLUE"),
    "B8_NIR":       (TM_MODALITY_S2L2A, "NIR_BROAD"),
    "B5":           (TM_MODALITY_S2L2A, "RED_EDGE_1"),
    "B6":           (TM_MODALITY_S2L2A, "RED_EDGE_2"),
    "B7":           (TM_MODALITY_S2L2A, "RED_EDGE_3"),
    "B8A":          (TM_MODALITY_S2L2A, "NIR_NARROW"),
    "B11":          (TM_MODALITY_S2L2A, "SWIR_1"),
    "B12":          (TM_MODALITY_S2L2A, "SWIR_2"),
    "VV_ascending": (TM_MODALITY_S1GRD, "VV"),
    "VH_ascending": (TM_MODALITY_S1GRD, "VH"),
    # *_enhanced -> TerraMind's dedicated RGB modality, not the S2L2A channels
    "B4_R_enhanced": (TM_MODALITY_IDENTITY, "RED"),
    "B3_G_enhanced": (TM_MODALITY_IDENTITY, "GREEN"),
    "B2_B_enhanced": (TM_MODALITY_IDENTITY, "BLUE"),
}

"""Canonical S2-ROSA band layout

`S2_BANDS` is the 20-band source
`S2-ROSA-V2` appends 3 CLAHE+gamma enhanced-RGB bands (21-23).
`*_RGB*` are the common index groups a model passes to the loaders / `rasterio.read`.
"""

S2_BANDS = (
    "B4_R", "B3_G", "B2_B", "B8_NIR", "B5", "B6", "B7", "B8A", "B11", "B12",
    "VV_ascending", "VH_ascending", "VV_descending", "VH_descending",
    "elevation", "slope", "aspect",
    "esa_urban_10m", "gisa_urban_10m", "wsf_urban_10m",
)
ENHANCED_RGB_BANDS = ("B4_R_enhanced", "B3_G_enhanced", "B2_B_enhanced")
S2_V2_BANDS = S2_BANDS + ENHANCED_RGB_BANDS

# Common index groups maping into the 23-band V2 imagery.
RGB = (1, 2, 3)                 # B4_R, B3_G, B2_B
S2_10M = (1, 2, 3, 4)           # RBG + B8_NIR
S2_20M = (5, 6, 7, 8, 9, 10)    # B5, B6, B7, B8A, B11, B12
S2_SAR = (11, 12, 13, 14)       # VV/VH ascending/descending
ENHANCED_RGB = (21, 22, 23)     # appended CLAHE+gamma enhanced RGB


DEFAULT_BANDS = RGB


def written_band_names(*, version=2):
    if version == 2:
        return list(S2_V2_BANDS)
    elif version == 1:
        return list(S2_BANDS)
    raise ValueError(f"Unknown version {version}, expected 1 or 2")



def read_window(src, bands, window):
    """Read ``(C, h, w)`` float32 from an open rasterio dataset, NaN/inf -> 0.

    The COG contains N/A values, convert to 0
    """
    arr = src.read(bands, window=window).astype("float32")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _standardize(img):
    """Per-image, per-channel standardisation of a ``(C, H, W)`` float array."""
    mean = img.mean(axis=(1, 2), keepdims=True)
    std = img.std(axis=(1, 2), keepdims=True) + 1e-6
    return (img - mean) / std


def apply_norm(img, bands, mean=None, std=None):
    """Standardise a ``(C, h, w)`` array (``C == len(bands)``).

    If no mean or std are given, standardise per-image. 
    Otherwise, use the given per-band mean/std.
    """
    if mean is None or std is None:
        return _standardize(img)
    idx = [b - 1 for b in bands]
    m = np.asarray(mean, dtype="float32")[idx].reshape(-1, 1, 1)
    s = np.asarray(std, dtype="float32")[idx].reshape(-1, 1, 1)
    s = np.where(s > 1e-6, s, 1.0)  # guard degenerate/constant bands
    return ((img - m) / s).astype("float32")

def terramind_norm_stats(band_indices):
    max_idx = max(BAND_INDEX_TO_NAME)
    mean = [0.0] * max_idx
    std = [1.0] * max_idx
    for i in band_indices:
        name = BAND_INDEX_TO_NAME[i]
        if name not in TM_STATS_KEY:
            raise KeyError(
                f"No TerraMind stats mapping for band '{name}' (index {i}); "
                f"add it to TM_STATS_KEY in consts.py."
            )
        modality, tm_band = TM_STATS_KEY[name]
        if modality == TM_MODALITY_IDENTITY:
            mean[i - 1] = 0.0
            std[i - 1] = 1.0
            continue
        pos = PRETRAINED_BANDS[modality].index(tm_band)
        scale = TM_RESCALE[modality]
        mean[i - 1] = v1_pretraining_mean[modality][pos] / scale
        std[i - 1] = v1_pretraining_std[modality][pos] / scale
    return mean, std

def _get_rgb_version(band_names):
    """band_names: list of resolved name strings (see _split_modalities)."""
    has_raw = any(n in ("B4_R", "B3_G", "B2_B") for n in band_names)
    has_enhanced = any(n in ("B4_R_enhanced", "B3_G_enhanced", "B2_B_enhanced") for n in band_names)

    if has_raw and has_enhanced:
        raise ValueError("Both raw and enhanced RGB bands present in self.bands; ambiguous rgb_source.")
    elif has_enhanced:
        rgb_source = "enhanced"
    elif has_raw:
        rgb_source = "raw"
    else:
        raise ValueError("No valid RGB source found in band names.")

    return {**RGB_SOURCES[rgb_source], **S2L2A_NON_RGB}


def split_modalities(img, band_indices):
    """
        Adapt datset images to dictionary format needed by TerraMind
        band_indices: self.bands, e.g. [4,5,6,7,8,9,10,11,12,13,14,21,22,23].
    """
    band_names = [BAND_INDEX_TO_NAME[i] for i in band_indices]
    s2l2a_bands = _get_rgb_version(band_names)
    idx = {name: i for i, name in enumerate(band_names)}
    s2 = img[[idx[col] for col in s2l2a_bands.values()]]
    s1 = img[[idx[col] for col in S1GRD_BANDS.values()]]
    return {
        "S2L2A": torch.from_numpy(np.ascontiguousarray(s2)),
        "S1GRD": torch.from_numpy(np.ascontiguousarray(s1)),
    }