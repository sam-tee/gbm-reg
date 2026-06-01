"""Small wrappers around ANTs image preprocessing operations."""
import ants
import numpy as np


def resample(
    image: ants.ANTsImage,
    spacing: tuple = (1, 1, 1),
    interp_type: int = 4,
) -> ants.ANTsImage:
    """Resample an ANTs image to isotropic or requested spacing.
    
    Args:
        image (ants.ANTsImage): Image volume to process.
        spacing (tuple): Voxel spacing associated with image data.
        interp_type (int): Optional interp type value. Defaults to `4`.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    return ants.resample_image(
        image, spacing, use_voxels=False, interp_type=interp_type
    )


def n4_bias_correct(image: ants.ANTsImage, shrink_factor: int = 2) -> ants.ANTsImage:
    """Apply N4 bias-field correction to an ANTs image.
    
    Args:
        image (ants.ANTsImage): Image volume to process.
        shrink_factor (int): Optional shrink factor value. Defaults to `2`.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    corrected = ants.n4_bias_field_correction(image, shrink_factor=shrink_factor)
    return corrected


def flip_image(image: ants.ANTsImage, axes_to_flip: tuple) -> ants.ANTsImage:
    """Flip an ANTs image along selected axes while preserving metadata.
    
    Args:
        image (ants.ANTsImage): Image volume to process.
        axes_to_flip (tuple): Axes to flip when reorienting an image.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    image_array = image.numpy()
    image_array = np.flip(image_array, axis=axes_to_flip)
    flipped_image = ants.from_numpy(
        image_array,
        origin=image.origin,
        spacing=image.spacing,
        direction=image.direction,
    )
    return flipped_image
