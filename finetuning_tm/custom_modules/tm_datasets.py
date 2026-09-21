import random
import typing
from pathlib import Path
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset
import albumentations as A

import torch.nn.functional as F
from torchgeo.datamodules.geo import BaseDataModule

from .norm import (
    DEFAULT_BANDS,
    read_window, apply_norm, split_modalities, terramind_norm_stats
)

"""window reads + per-image standardisation"""
import numpy as np

def _read_split_csv(dataset_dir, split):
    csv = Path(dataset_dir) / "splits" / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv}")
    return pd.read_csv(csv)

def _resize_pair(img, mask, target_size):
    """img: (C,H,W) float32 array, mask: (H,W) float32 array."""
    img_t = torch.from_numpy(img).unsqueeze(0)          # (1,C,H,W)
    img_t = F.interpolate(img_t, size=(target_size, target_size),
                           mode="bilinear", align_corners=False)
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    mask_t = F.interpolate(mask_t, size=(target_size, target_size),
                            mode="nearest")
    return img_t.squeeze(0).numpy(), mask_t.squeeze(0).squeeze(0).numpy()

def _remap_mask_paths(df, dataset_dir, mask_dirname):
    """Point ``mask_path`` at an alternative label set living beside the
    pipeline's ``masks_raster/`` (e.g. ``mask_osm_10`` written by
    OpenStreetMapTest/dataset_hr_masks.py --scale 1). ``None`` keeps the CSV's
    masks unchanged. The alternative masks are rasterised on each tile's own
    grid, so dims/CRS match; every remapped file must exist — missing labels
    are an error, not a silent filter, so label-source comparisons stay on
    identical tile sets."""
    if mask_dirname is None:
        return df
    df = df.copy()
    df["mask_path"] = df["mask_path"].map(
        lambda rel: str(Path(rel).parent.parent / mask_dirname / Path(rel).name))
    missing = [rel for rel in df["mask_path"]
               if not (Path(dataset_dir) / rel).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(df)} tiles have no mask under "
            f"<split>/{mask_dirname}/ (first: {missing[0]}). Generate them "
            f"with OpenStreetMapTest/dataset_hr_masks.py --scale 1 "
            f"--out-dirname {mask_dirname}."
        )
    return df

class RoadTileDataset(Dataset):
    """Use this dataset for training. Random pixel crops from the 512x512 train tiles.

    Length by default is 10 * len(train_tiles)
    """

    def __init__(self, dataset_dir, bands=DEFAULT_BANDS, image_size=256,
                 length=None, normalize=True, norm_mean=None, norm_std=None,
                 mask_dirname=None, target_size=None, transform: A.BasicTransform | None = None,
                 modality_output: str = "split"
                ):
        
        self.dataset_dir = Path(dataset_dir)
        self.df = _remap_mask_paths(
            _read_split_csv(dataset_dir, "train"), dataset_dir, mask_dirname
        ).reset_index(drop=True)
        self.bands = list(bands)
        self.image_size = image_size
        self.normalize = normalize
        self.norm_mean = norm_mean  # frozen train stats (or None -> per-image)
        self.norm_std = norm_std
        self.length = length if length is not None else 10 * len(self.df)
        self.target_size = target_size   # optional post-crop resize
        self.transform = transform
        self.modality_output = modality_output 

        if self.norm_mean is None or self.norm_std is None:
            raise ValueError("No frozen train stats given")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Random crop and normalise"""
        row = self.df.iloc[random.randrange(len(self.df))]
        s = self.image_size
        with rasterio.open(self.dataset_dir / row["image_path"]) as src, \
                rasterio.open(self.dataset_dir / row["mask_path"]) as msrc:
            H, W = src.height, src.width
            top = random.randint(0, max(0, H - s))
            left = random.randint(0, max(0, W - s))
            win = Window(left, top, min(s, W - left), min(s, H - top))
            img = read_window(src, self.bands, win)                 # (C, h, w)
            mask = (msrc.read(1, window=win) > 0).astype("float32")  # (h, w)

        if self.normalize:
            img = apply_norm(img, self.bands, self.norm_mean, self.norm_std)

        c, h, w = img.shape
        if (h, w) != (s, s):  # short edge tile -> zero-pad (train tiles are 512, rare)
            pad_i = np.zeros((c, s, s), dtype="float32"); pad_i[:, :h, :w] = img
            pad_m = np.zeros((s, s), dtype="float32"); pad_m[:h, :w] = mask
            img, mask = pad_i, pad_m

        if self.target_size is not None:               
            img, mask = _resize_pair(img, mask, self.target_size)

        if self.transform is not None:
            out = self.transform(
                image=np.ascontiguousarray(img.transpose(1, 2, 0)),   # CHW -> HWC
                mask=mask,
            )
            img = np.ascontiguousarray(out["image"].transpose(2, 0, 1))   # HWC -> CHW
            mask = out["mask"]

        if self.modality_output == "split":
            image = split_modalities(img, self.bands)   # {"S2L2A": ..., "S1GRD": ...}
        elif self.modality_output == "stacked":
            image = torch.from_numpy(np.ascontiguousarray(img)).float()   # (C, H, W) tensor
        else:
            raise ValueError(f"Unknown modality_output: {self.modality_output!r}")
        
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()   # (H, W), Long — see (4)
        # return {
        #     "image": image,
        #     "mask": mask,
        #     "filename": f"{row['zone_name']}_{top}_{left}.png",  # or _q{quad} in TileCropDataset
        # }
        return {
            "image": image,
            "mask": mask,
            "filename": str(self.dataset_dir / row["image_path"]),  # terratorch's contract: real path
        }


class TileCropDataset(Dataset):
    """
    Use this dataset for validation/testing. 

    Deterministic 2x2 crops from the 512x512 tiles - sliding window with no overlaps
    Item ``idx`` -> tile ``idx // 4``, quadrant ``idx % 4`` (row-major)
    """

    def __init__(self, dataset_dir, split, bands=DEFAULT_BANDS,
                 normalize=True, norm_mean=None, norm_std=None,
                  mask_dirname=None, target_size=None,
                  modality_output: str = "split"
                ):
        self.dataset_dir = Path(dataset_dir)
        self.df = _remap_mask_paths(
            _read_split_csv(dataset_dir, split), dataset_dir, mask_dirname
        ).reset_index(drop=True)
        self.bands = list(bands)
        self.image_size = 256
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.per_tile = 4  # Assume 512x512 tiles, hence 4 patches.
        self.target_size = target_size
        self.modality_output = modality_output

        if self.norm_mean is None or self.norm_std is None:
            raise ValueError("No frozen train stats given")

    def __len__(self):
        return len(self.df) * self.per_tile

    def __getitem__(self, idx):
        """Sliding Window and normalise"""
        row = self.df.iloc[idx // self.per_tile]
        quad = idx % self.per_tile
        s = self.image_size
        top = (quad // 2) * s
        left = (quad % 2) * s
        with rasterio.open(self.dataset_dir / row["image_path"]) as src, \
                rasterio.open(self.dataset_dir / row["mask_path"]) as msrc:
            win = Window(left, top, s, s)
            img = read_window(src, self.bands, win)                  # (C, s, s)
            mask = (msrc.read(1, window=win) > 0).astype("float32")  # (s, s)

        if self.target_size is not None:                # <-- new
            img, mask = _resize_pair(img, mask, self.target_size)

        if self.normalize:
            img = apply_norm(img, self.bands, self.norm_mean, self.norm_std)
            
        if self.modality_output == "split":
            image = split_modalities(img, self.bands)   # {"S2L2A": ..., "S1GRD": ...}
        elif self.modality_output == "stacked":
            image = torch.from_numpy(np.ascontiguousarray(img)).float()
        else:
            raise ValueError(f"Unknown modality_output: {self.modality_output!r}")
        
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()   #

        return {
            "image": image,
            "mask": mask,
            "filename": str(self.dataset_dir / row["image_path"]),  # terratorch's contract: real path
        }

class WholeTileDataset(Dataset):
    """Use this dataset for Predict: one item per tile, no cropping.

    Windowing is delegated to terratorch's ``tiled_inference`` (set
    ``tiled_inference_parameters`` on the task). That keeps ``filename``
    pointing at a raster whose dimensions match the prediction, which is what
    ``terratorch.cli_tools.save_prediction`` assumes when it reopens the source
    to copy CRS/transform/nodata.
    """

    def __init__(self, dataset_dir, split, bands=DEFAULT_BANDS,
                 normalize=True, norm_mean=None, norm_std=None,
                 mask_dirname=None, target_size=None,
                 modality_output: str = "stacked"
                ):
        self.dataset_dir = Path(dataset_dir)
        self.df = _remap_mask_paths(
            _read_split_csv(dataset_dir, split), dataset_dir, mask_dirname
        ).reset_index(drop=True)
        self.bands = list(bands)
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.target_size = target_size
        self.modality_output = modality_output 

        if self.norm_mean is None or self.norm_std is None:
            raise ValueError("No frozen train stats given")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.dataset_dir / row["image_path"]
        with rasterio.open(image_path) as src, \
                rasterio.open(self.dataset_dir / row["mask_path"]) as msrc:
            win = Window(0, 0, src.width, src.height)        # full tile, no crop
            img = read_window(src, self.bands, win)          # (C, H, W)
            mask = (msrc.read(1) > 0).astype("float32")      # (H, W)

        if self.target_size is not None:
            img, mask = _resize_pair(img, mask, self.target_size)

        if self.normalize:
            img = apply_norm(img, self.bands, self.norm_mean, self.norm_std)

        if self.modality_output == "split":
            image = split_modalities(img, self.bands)
        elif self.modality_output == "stacked":
            image = torch.from_numpy(np.ascontiguousarray(img)).float()
        else:
            raise ValueError(f"Unknown modality_output: {self.modality_output!r}")
        
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return {
            "image": image,
            "mask": mask,
            "filename": str(image_path),
        }

class RoadDataModule(BaseDataModule):
    """
    Train - random native crops; 
    val/test - deterministic 2x2 quadrant crops.
    prediction - wjole tile no crop 

    ``mask_dirname`` switches the label source: ``None`` (default) uses the
    pipeline masks the split CSVs point at (``masks_raster/``, e.g. CDNGI);
    a dir name (e.g. ``mask_osm_10``) uses the alternative masks stored
    beside them, applied to train/val/test alike. To cross-evaluate (train on
    one label set, test on the other), pass a different ``--data.mask_dirname``
    to ``unet.cli test``.
    """

    def __init__(self, dataset_dir: str, bands: tuple[int, ...] = DEFAULT_BANDS,
                 batch_size: int = 16, num_workers: int = 2, image_size: int = 256,
                 length: int | None = None, normalize: bool = True, norm_source: str = "manual",
                 norm_mean: list[float] | None = None, norm_std: list[float] | None = None,
                 mask_dirname: str | None = None, target_size: int | None = None,
                 predict_split: str = "test", predict_batch_size: int | None = None, 
                 transform_train: bool = False, modality_output: str = "split"):

        # pass it and ignore it — you construct datasets inside your dataloader methods anyway.
        # Keep it out ofyour own signature so jsonargparse never has to resolve type[Dataset[Sample]].
        super().__init__(dataset_class=RoadTileDataset, batch_size=batch_size, num_workers=num_workers)

        self.dataset_dir = Path(dataset_dir)
        self.bands = tuple(bands)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.length = length
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.mask_dirname = mask_dirname
        self.target_size = target_size
        self.transform_train = transform_train
        self.modality_output = modality_output

        self.predict_split = predict_split
        # whole 512 tiles are ~4x the pixels of a 256 crop
        self.predict_batch_size = predict_batch_size or max(1, batch_size // 4)

        self.norm_source = norm_source
        if norm_source == "terramind":
            self.norm_mean, self.norm_std = terramind_norm_stats(self.bands)
        elif norm_source == "manual":
            if norm_mean is None or norm_std is None:
                raise ValueError("norm_source='manual' requires norm_mean and norm_std")
        else:
            raise ValueError(f"Unknown norm_source: {norm_source!r}")

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.

        One small addition now that the paste shows _valid_attribute's raise path: 
        if you ever did want a real on-device aug later (e.g. flips), 
        set self.train_aug only and keep val_aug/test_aug as identity-like callables rather than deleting aug 
        — otherwise the fallback chain raises MisconfigurationException for the splits you left unset.
          For now, the flat no-op sidesteps all of it.
        """
        return batch

    def train_dataloader(self):
        if self.transform_train:
            transform = A.D4(p=1.0)
        else:
            transform = None
            
        ds = RoadTileDataset(
            self.dataset_dir, self.bands, self.image_size, self.length, self.normalize,
            self.norm_mean, self.norm_std, self.mask_dirname, self.target_size, transform=transform,
            modality_output=self.modality_output
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,  # randomness is in __getitem__; DDP adds DistributedSampler
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
    
    def _eval_loader(self, split):
        ds = TileCropDataset(
            self.dataset_dir, split, self.bands, self.normalize,
            self.norm_mean, self.norm_std, self.mask_dirname, self.target_size,
            modality_output=self.modality_output
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self):
        return self._eval_loader("val")

    def test_dataloader(self):
        return self._eval_loader("test")
    
    def predict_dataloader(self):
        ds = WholeTileDataset(
            self.dataset_dir, self.predict_split, self.bands, self.normalize,
            self.norm_mean, self.norm_std, self.mask_dirname, self.target_size,
            modality_output=self.modality_output
        )
        return DataLoader(
            ds,
            batch_size=self.predict_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )
