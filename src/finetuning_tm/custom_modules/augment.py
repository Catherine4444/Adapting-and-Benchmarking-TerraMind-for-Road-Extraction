import albumentations as A

_D4_NAMES = {
    "e": "identity", "r90": "rot90", "r180": "rot180", "r270": "rot270",
    "v": "vflip", "h": "hflip", "t": "transpose", "hvt": "antitranspose",
}

def build_transform(flip: bool = True, sharpen: bool = False, noise: bool = False,
                    blur: bool = False, colour: bool = False,
                    p: float = 0.5, seed: int | None = None):
    """Build the Albumentations pipeline from a set of augmentation toggles.

    `flip` is the D4 dihedral group via `A.D4`: a single transform that picks a
    *uniformly* random element of the 8 flip/rotation symmetries (identity, 3
    rotations, 2 axis flips, 2 diagonal flips) — cleaner and unbiased vs.
    composing HFlip/VFlip/RandomRotate90/Transpose, and it subsumes the "50%
    horizontal/vertical flip" idea. The photometric toggles each fire with
    probability `p`. Magnitudes are tuned for z-scored input (see module
    docstring). Albumentations is imported lazily.
    """

    tfms = []
    if flip:
        tfms.append(A.D4(p=1.0))
    if sharpen:
        tfms.append(A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=p))
    if noise:
        # std in normalised units: ~5-20% of the unit std of the z-scored bands.
        tfms.append(A.GaussNoise(std_range=(0.05, 0.2), per_channel=True, p=p))
    if blur:
        tfms.append(A.GaussianBlur(blur_limit=(3, 5), p=p))
    if colour:
        tfms.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p))
    if not tfms:
        raise ValueError("build_transform: no augmentations enabled (all toggles False)")
    return A.Compose(tfms, seed=seed)