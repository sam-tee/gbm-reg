"""Shared segmentation label mappings and helpers."""

import ants
import numpy as np

GREY_MATTER_LABELS = {
    3,
    8,
    10,
    11,
    12,
    13,
    17,
    18,
    26,
    28,
    42,
    47,
    49,
    50,
    51,
    52,
    53,
    54,
    58,
    60,
}

WHITE_MATTER_LABELS = {
    2,
    7,
    41,
    46,
}


def remap_freesurfer_labels(freesurfer_label_image: ants.ANTsImage) -> ants.ANTsImage:
    """Remap FreeSurfer/SynthSeg labels to compact tissue labels.

    Args:
        freesurfer_label_image (ants.ANTsImage): FreeSurfer or SynthSeg label image.

    Returns:
        ants.ANTsImage: ANTs image with `0` for background, `1` for white matter,
        and `2` for grey matter.
    """
    labels_np = freesurfer_label_image.numpy()
    remapped_data = np.zeros_like(labels_np, dtype="uint8")
    remapped_data[np.isin(labels_np, list(WHITE_MATTER_LABELS))] = 1
    remapped_data[np.isin(labels_np, list(GREY_MATTER_LABELS))] = 2
    return ants.new_image_like(freesurfer_label_image, remapped_data)
