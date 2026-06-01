"""Mask generation routines for in-vivo and ex-vivo brain images."""
import subprocess
from pathlib import Path

import ants
import numpy as np

from src.classes import PreprocessConfig


def exvivo_mask(image: ants.ANTsImage, config: PreprocessConfig) -> ants.ANTsImage:
    """Build an ex-vivo brain mask using the configured masking method.
    
    Args:
        image (ants.ANTsImage): Image volume to process.
        config (PreprocessConfig): Configuration object or dictionary controlling the operation.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    mask_algo = config.tissue_mask_algorithm.lower()
    if mask_algo == "gmm":
        mask = gmm_tissue_mask(image)
    elif mask_algo == "otsu":
        mask = otsu_tissue_mask(image)
    else:
        print(
            f"Error: tissue_mask_algorithm must be one of [gmm, otsu] not {mask_algo}"
        )
        print("Using complete image")
        mask = ants.new_image_like(image, np.ones(image.shape))
    return mask


def skull_strip(
    image: ants.ANTsImage, temp_dir: Path, config: PreprocessConfig
) -> ants.ANTsImage:
    """Create a brain mask for an in-vivo scan using the configured backend.
    
    Args:
        image (ants.ANTsImage): Image volume to process.
        temp_dir (Path): Directory path used by the operation.
        config (PreprocessConfig): Configuration object or dictionary controlling the operation.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    algorithm = config.skull_strip_algorithm.lower()
    if algorithm == "hd-bet":
        brain_mask = run_hd_bet(image, temp_dir, config.hd_bet_flags)
    elif algorithm == "fsl-bet":
        brain_mask = run_fsl_bet(image, temp_dir, config.fsl_bet_threshold)
    else:
        print("Error: Skull stripping algorithm is not one of [hd-bet, fsl-bet]")
        print("Returning mask of all ones")
        brain_mask = ants.new_image_like(image, np.ones(image.shape))
    return brain_mask


def run_fsl_bet(
    image: ants.ANTsImage, temp_dir: Path, threshold: float = 0.5
) -> ants.ANTsImage:
    """Uses FSL_BET to generate brain mask
    
    Args:
        image (ants.ANTsImage): Image volume to process.
        temp_dir (Path): Directory path used by the operation.
        threshold (float): Optional threshold value. Defaults to `0.5`.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    pre_file = temp_dir / "pre_bet.nii.gz"
    post_file = temp_dir / "post_bet.nii.gz"
    pre_file.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(image, str(pre_file))
    cmd = ["bet", pre_file, post_file, "-f", str(threshold), "-m", "-B"]
    try:
        subprocess.run(cmd, text=True, check=True, capture_output=True)
        image = ants.image_read(str(post_file))
    except subprocess.CalledProcessError as e:
        print("FSL-BET called error - using input image")
        print(e)
    pre_file.unlink()
    post_file.unlink(missing_ok=True)
    return image


def run_hd_bet(
    image: ants.ANTsImage, temp_dir: Path, options: list[str] | None = None
) -> ants.ANTsImage:
    """Uses HD-BET to generate mask for the brain
    
    Args:
        image (ants.ANTsImage): Image volume to process.
        temp_dir (Path): Directory path used by the operation.
        options (list[str] | None): Optional options value. Defaults to `None`.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    pre_file = temp_dir / "t1c.nii.gz"
    post_file = temp_dir / "stripped.nii.gz"
    mask_file = temp_dir / "stripped_bet.nii.gz"
    pre_file.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(image, str(pre_file))
    cmd = ["hd-bet", "-i", pre_file, "-o", post_file, "--save_bet_mask"]
    if options is not None:
        cmd.extend(options)
    try:
        subprocess.run(cmd, text=True, check=True, capture_output=True)
        mask = ants.image_read(str(mask_file))
    except subprocess.CalledProcessError as e:
        print(f"Error: HD-BET called error {e}")
        print("USing full image")
        mask = ants.new_image_like(image, np.ones(image.shape))
    pre_file.unlink()
    post_file.unlink(missing_ok=True)
    mask_file.unlink(missing_ok=True)
    return mask


def otsu_tissue_mask(image: ants.ANTsImage) -> ants.ANTsImage:
    """Create a tissue mask from Otsu thresholding.
    
    Args:
        image (ants.ANTsImage): Image volume to process.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    return ants.get_mask(image, cleanup=2)


def gmm_tissue_mask(image: ants.ANTsImage) -> ants.ANTsImage:
    """Uses atropos with one class to get tissue mask
    
    Args:
        image (ants.ANTsImage): Image volume to process.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    x = ants.new_image_like(image, np.ones(image.shape))
    mask = ants.atropos(
        a=image,
        x=x,
        i="KMeans[1]",
        m="[0.1,1x1x1]",
        c="[5,0]",
    )["segmentation"]
    return mask
