"""
Creates virtual slices at regular steps over the ROI based on segmentation mask
Then registers the virtual slices to given slice
"""

from pathlib import Path

import ants
import numpy as np


def get_bbox(labels: ants.ANTsImage) -> tuple:
    """Returns tuple of (min_coords, max_coords) of non-zero parts of labels image
    
    Args:
        labels (ants.ANTsImage): Label image or label values used by the operation.
    
    Returns:
        tuple: Tuple containing the values described by the return annotation.
        """
    mask_np = labels.numpy()
    coords = np.argwhere(mask_np > 0)
    return *coords.min(axis=0), *coords.max(axis=0)


def create_slices(
    roi: ants.ANTsImage,
    bbox: tuple,
    slice_dir: Path = Path("./slices"),
    num_steps: int = 20,
    thickness_mm: float = 15.0,
):
    """Creates virtual slices in given step from min_y to max_y
    
    Args:
        roi (ants.ANTsImage): Roi value used by the operation.
        bbox (tuple): Bbox value used by the operation.
        slice_dir (Path): Directory path used by the operation.
        num_steps (int): Optional num steps value. Defaults to `20`.
        thickness_mm (float): Optional thickness mm value. Defaults to `15.0`.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    slice_dir.mkdir(exist_ok=True, parents=True)
    _, y_min, _, _, y_max, _ = bbox
    y_positions = np.linspace(y_min, y_max, num_steps)

    y_spacing = ants.get_spacing(roi)[1]
    thickness_voxels = thickness_mm / y_spacing

    for i, center_y in enumerate(y_positions):
        # Define the half-thickness boundaries
        start_y = center_y - (thickness_voxels / 2)
        end_y = center_y + (thickness_voxels / 2)
        lower_ind = [0, int(start_y), 0]
        upper_ind = [roi.shape[0], int(end_y), roi.shape[2]]
        virtual_slice = ants.crop_indices(roi, lower_ind, upper_ind)
        slice_filename = slice_dir / f"slice_{i:03d}_y{int(center_y)}.nii.gz"
        ants.image_write(virtual_slice, str(slice_filename))


def get_cerebrum_mask(
    seg_labels: ants.ANTsImage, hemisphere: str = "right"
) -> ants.ANTsImage:
    """Creates a mask of the left or right cerebrum based on freesurfer labels
    
    Args:
        seg_labels (ants.ANTsImage): Segmentation label image used to identify anatomy.
        hemisphere (str): Optional hemisphere value. Defaults to `'right'`.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    left_cerebrum_labels = [2, 3, 10, 11, 12, 13, 17, 18, 26, 28]
    right_cerebrum_labels = [41, 42, 49, 50, 51, 52, 53, 54, 58, 60]
    data = seg_labels.numpy()
    if hemisphere == "right":
        mask_data = np.isin(data, right_cerebrum_labels).astype(np.float32)
    else:
        mask_data = np.isin(data, left_cerebrum_labels).astype(np.float32)
    return ants.new_image_like(seg_labels, mask_data)


def get_roi(
    tumour_labels: ants.ANTsImage,
    segmentation_labels: ants.ANTsImage,
    full_brain: ants.ANTsImage,
) -> ants.ANTsImage:
    """Takes in tumour labels and FreeSurfer segmentation labels and returns the full brain masked
    
    Args:
        tumour_labels (ants.ANTsImage): Tumour label image used to localise the tumour region.
        segmentation_labels (ants.ANTsImage): Label data used by the operation.
        full_brain (ants.ANTsImage): Full brain value used by the operation.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    tumour_bbox = get_bbox(tumour_labels)
    tumour_x_min, _, _, tumour_x_max, _, _ = tumour_bbox

    tumour_middle_x = (tumour_x_min + tumour_x_max) / 2

    brain_bbox = get_bbox(segmentation_labels)
    brain_x_min, _, _, brain_x_max, _, _ = brain_bbox
    brain_middle_x = (brain_x_min + brain_x_max) / 2

    if tumour_middle_x > brain_middle_x:
        hemi = "left"
    else:
        hemi = "right"
    mask = get_cerebrum_mask(segmentation_labels, hemi)
    masked_brain = ants.mask_image(full_brain, mask)
    ants.image_write(masked_brain, "test_brain.nii.gz")
    return masked_brain


def register_slab_to_slices(slab: ants.ANTsImage, slice_dir: Path, output_dir: Path):
    """Does an affine reigstration using MI as metric on each slice in slice dir
    
    Args:
        slab (ants.ANTsImage): Slab identifier or slab image to process.
        slice_dir (Path): Directory path used by the operation.
        output_dir (Path): Directory where generated outputs are written.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    output_dir.mkdir(exist_ok=True, parents=True)
    com_slab = ants.get_center_of_mass(slab)
    for slice_path in slice_dir.glob("*.nii.gz"):
        slice_img = ants.image_read(str(slice_path))
        com_slice = ants.get_center_of_mass(slice_img)

        init_tx = ants.create_ants_transform(
            transform_type="Euler3DTransform",
            translation=np.array(com_slab) - np.array(com_slice),
        )
        tmp_file = output_dir / "tmp.mat"
        ants.write_transform(init_tx, str(tmp_file))
        try:
            reg = ants.registration(
                fixed=slice_img,
                moving=slab,
                type_of_transform="SyN",
                metric="MI",
                initial_transform=[str(tmp_file)],
            )
        except RuntimeError:
            continue
        warped_slab = reg["warpedmovout"]
        ants.image_write(warped_slab, str(output_dir / f"warped_{slice_path.name}"))
        metric = ants.image_mutual_information(slice_img, warped_slab)
        print(f"Registered {slice_path.name} with MI: {metric}")
        tmp_file.unlink()


if __name__ == "__main__":
    tumour_labels = ants.image_read("./IM008/PreM/seg.nii.gz")
    tumour_labels = ants.reorient_image2(tumour_labels, "RAS")
    seg_labels = ants.image_read("./IM008/PreM/freesurfer.nii.gz")
    seg_labels = ants.reorient_image2(seg_labels, "RAS")
    bbox = get_bbox(tumour_labels)
    full_brain = ants.image_read("./IM008/PreM/t1.nii.gz")
    full_brain = ants.reorient_image2(full_brain, "RAS")
    brain_roi = get_roi(tumour_labels, seg_labels, full_brain)
    create_slices(brain_roi, bbox)
    slab = ants.image_read("./IM008/PostM/t2.nii.gz")
    register_slab_to_slices(slab, Path("./slices"), Path("./reg"))
