"""Intensity normalisation routines for ANTs images."""
import ants
import numpy as np

from src.classes import PreprocessConfig


def normalise(
    image: ants.ANTsImage, mask: ants.ANTsImage, config: PreprocessConfig
) -> ants.ANTsImage:
    """Normalise an ANTs image using the configured normalisation strategy.
    
    Args:
        image (ants.ANTsImage): Image volume to process.
        mask (ants.ANTsImage): Binary or label mask used to limit the operation.
        config (PreprocessConfig): Configuration object or dictionary controlling the operation.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    norm_method = config.normalisation_method.lower()
    if norm_method == "z-score":
        image_norm = z_normalise(image, mask)
    elif norm_method == "min-max":
        image_norm = min_max_normalise(
            image, mask, config.min_intensity, config.max_intensity
        )
    else:
        print(
            f"Error: Normalisation method must be one of [z-score, min-max] not {norm_method}"
        )
        print("Using un-normalised image")
        image_norm = image
    return image_norm


def z_normalise(image: ants.ANTsImage, mask: ants.ANTsImage) -> ants.ANTsImage:
    """Apply masked z-score intensity normalisation.
    
    Args:
        image (ants.ANTsImage): Image volume to process.
        mask (ants.ANTsImage): Binary or label mask used to limit the operation.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    image_np, np_mask = image.numpy(), mask.numpy()
    roi = image_np[np_mask > 0]
    if roi.size > 0:
        mean = roi.mean()
        std = roi.std()
        norm_data = np.where(np_mask > 0, (image_np - mean) / (std + 1e-8), 0)
    else:
        print("ROI has zero size - aborting normalisation")
        norm_data = image_np
    return ants.new_image_like(image, norm_data.astype(np.float32))


def min_max_normalise(
    image: ants.ANTsImage, mask: ants.ANTsImage, min: int, max: int
) -> ants.ANTsImage:
    """Returns an image with intensities in mask normalised to [min,max]
    
    Args:
        image (ants.ANTsImage): Image volume to process.
        mask (ants.ANTsImage): Binary or label mask used to limit the operation.
        min (int): Min value used by the operation.
        max (int): Max value used by the operation.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    image_np, np_mask = image.numpy(), mask.numpy()
    roi = image_np[np_mask > 0]
    if roi.size > 0:
        roi_min, roi_max = roi.min(), roi.max()
        norm_data = np.where(
            np_mask > 0,
            (image_np - roi_min) / (roi_max - roi_min + 1e-8) * (max - min) + min,
            0,
        )
    else:
        print("ROI has zero size - aborting normalisation")
        norm_data = image_np
    return ants.new_image_like(image, norm_data.astype(np.float32))
