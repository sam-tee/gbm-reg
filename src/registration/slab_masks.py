"""Mask construction and spatial projection helpers for slab registration."""
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def nib_world_bounds(
    img: nib.Nifti1Image, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the world-coordinate bounds of a nibabel image."""
    if mask is None:
        ijk = (
            np.array(np.meshgrid(*[[0, s - 1] for s in img.shape[:3]], indexing="ij"))
            .reshape(3, -1)
            .T
        )
    else:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return nib_world_bounds(img)
        ijk = coords
    xyz = nib.affines.apply_affine(img.affine, ijk)
    return xyz.min(axis=0), xyz.max(axis=0)


def set_image_center_world(
    src: nib.Nifti1Image, center_xyz: np.ndarray
) -> nib.Nifti1Image:
    """Update an image affine so its centre is at a requested world coordinate."""
    shape = np.asarray(src.shape[:3], dtype=float)
    voxel_center = (shape - 1.0) / 2.0
    affine = src.affine.copy()
    affine[:3, 3] = center_xyz - affine[:3, :3] @ voxel_center
    return nib.Nifti1Image(np.asanyarray(src.dataobj), affine, src.header)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    labeled, n_labels = ndimage.label(mask, structure=structure)
    if n_labels == 0:
        return mask.astype(bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == int(counts.argmax())


def tumour_label_mask(tumour_path: Path, out_path: Path, label: int = 2) -> Path:
    """Save a binary mask for one tumour label."""
    img = nib.load(tumour_path)
    data = np.asanyarray(img.dataobj)
    mask = largest_connected_component(data == label).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), out_path)
    return out_path


def oedema_ap_range_brain_mask(
    tumour_path: Path, brain_mask_path: Path, out_path: Path, label: int = 2
) -> tuple[Path, float, float]:
    """Save a brain mask restricted to the A-P range of one tumour label."""
    tumour_img = nib.load(tumour_path)
    brain_img = nib.load(brain_mask_path)
    tumour = np.asanyarray(tumour_img.dataobj)
    oedema = largest_connected_component(tumour == label)
    if not np.any(oedema):
        raise ValueError(
            f"No voxels found for tumour/oedema label {label} in {tumour_path}"
        )

    oedema_world = nib.affines.apply_affine(tumour_img.affine, np.argwhere(oedema))
    ap_min = float(oedema_world[:, 1].min())
    ap_max = float(oedema_world[:, 1].max())

    brain = np.asanyarray(brain_img.dataobj) > 0
    coords = np.argwhere(brain)
    world = nib.affines.apply_affine(brain_img.affine, coords)
    in_ap_band = (world[:, 1] >= ap_min) & (world[:, 1] <= ap_max)

    out = np.zeros(brain.shape, dtype=np.uint8)
    selected = coords[in_ap_band]
    out[selected[:, 0], selected[:, 1], selected[:, 2]] = 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, brain_img.affine, brain_img.header), out_path)
    return out_path, ap_min, ap_max


def patient_ap_positions(tumour_path: Path, step_mm: float) -> np.ndarray:
    """Generate anterior-posterior candidate positions from tumour extent."""
    tumour = nib.load(tumour_path)
    data = np.asanyarray(tumour.dataobj)
    mask = largest_connected_component(data == 2)
    if not np.any(mask):
        mask = data > 0
    coords = np.argwhere(mask)
    world = nib.affines.apply_affine(tumour.affine, coords)
    y_min, y_max = np.floor(world[:, 1].min()), np.ceil(world[:, 1].max())
    positions = np.arange(y_min, y_max + (0.5 * step_mm), step_mm)
    if positions[-1] < y_max:
        positions = np.append(positions, y_max)
    return positions.astype(float)


def nearest_resample_mask_to_grid(
    mask_img: nib.Nifti1Image, fixed_img: nib.Nifti1Image
) -> np.ndarray:
    """Nearest-neighbour resample a mask onto another image grid."""
    mask = np.asanyarray(mask_img.dataobj) > 0
    fixed_shape = fixed_img.shape[:3]
    grid = np.indices(fixed_shape, dtype=np.float32).reshape(3, -1).T
    world = nib.affines.apply_affine(fixed_img.affine, grid)
    moving_ijk = nib.affines.apply_affine(np.linalg.inv(mask_img.affine), world)
    moving_ijk = np.rint(moving_ijk).astype(np.int64)
    valid = np.all((moving_ijk >= 0) & (moving_ijk < np.asarray(mask.shape)), axis=1)
    out = np.zeros(grid.shape[0], dtype=np.uint8)
    idx = moving_ijk[valid]
    out[valid] = mask[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.uint8)
    return out.reshape(fixed_shape)


def project_post_roi_to_pre(
    post_roi_img: nib.Nifti1Image, pre_img: nib.Nifti1Image
) -> np.ndarray:
    """Project a post-mortem ROI mask into pre-mortem image space."""
    roi = np.asanyarray(post_roi_img.dataobj) > 0
    out = np.zeros(pre_img.shape[:3], dtype=np.uint8)
    coords = np.argwhere(roi)
    if coords.size == 0:
        return out
    world = nib.affines.apply_affine(post_roi_img.affine, coords)
    pre_ijk = np.rint(
        nib.affines.apply_affine(np.linalg.inv(pre_img.affine), world)
    ).astype(np.int64)
    valid = np.all((pre_ijk >= 0) & (pre_ijk < np.asarray(out.shape)), axis=1)
    idx = pre_ijk[valid]
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return out


def save_roi_masks(
    post_t2_path: Path,
    pre_t1c_path: Path,
    pre_oedema_path: Path,
    out_post: Path,
    out_pre: Path,
) -> tuple[Path, Path, int]:
    """Save post-space and pre-space ROI masks for one initialized slab."""
    post_img = nib.load(post_t2_path)
    pre_img = nib.load(pre_t1c_path)
    oedema_img = nib.load(pre_oedema_path)
    post_data = np.asanyarray(post_img.dataobj)
    roi = nearest_resample_mask_to_grid(oedema_img, post_img)
    roi = (roi & (post_data != 0)).astype(np.uint8)
    post_roi_img = nib.Nifti1Image(roi, post_img.affine, post_img.header)
    nib.save(post_roi_img, out_post)

    pre_roi = project_post_roi_to_pre(post_roi_img, pre_img)
    nib.save(nib.Nifti1Image(pre_roi, pre_img.affine, pre_img.header), out_pre)
    return out_post, out_pre, int(roi.sum())


def hemisphere_center_x(pre_mask_path: Path, tumour_path: Path) -> float:
    """Compute the x-coordinate centre of the tumour-side hemisphere."""
    brain = nib.load(pre_mask_path)
    tumour = nib.load(tumour_path)
    brain_data = np.asanyarray(brain.dataobj) > 0
    tumour_data = np.asanyarray(tumour.dataobj) == 2
    if not np.any(tumour_data):
        tumour_data = np.asanyarray(tumour.dataobj) > 0

    brain_world = nib.affines.apply_affine(brain.affine, np.argwhere(brain_data))
    tumour_world = nib.affines.apply_affine(tumour.affine, np.argwhere(tumour_data))
    mid = 0.5 * (brain_world[:, 0].min() + brain_world[:, 0].max())
    tumour_x = float(tumour_world[:, 0].mean())
    hemi = brain_world[:, 0] >= mid if tumour_x >= mid else brain_world[:, 0] < mid
    return float(0.5 * (brain_world[hemi, 0].min() + brain_world[hemi, 0].max()))
