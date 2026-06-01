"""Optimisation-based slab registration routines."""
from pathlib import Path

import ants
import numpy as np
from scipy import ndimage, optimize


def _ants_indices_to_physical(image, indices: np.ndarray) -> np.ndarray:
    """Convert ANTs voxel indices to physical coordinates."""
    origin = np.asarray(image.origin, dtype=float)
    spacing = np.asarray(image.spacing, dtype=float)
    direction = np.asarray(image.direction, dtype=float)
    return origin + (indices * spacing) @ direction.T


def _ants_physical_to_indices(image, points: np.ndarray) -> np.ndarray:
    """Convert physical coordinates to ANTs voxel indices."""
    origin = np.asarray(image.origin, dtype=float)
    spacing = np.asarray(image.spacing, dtype=float)
    direction = np.asarray(image.direction, dtype=float)
    return ((points - origin) @ direction) / spacing


def _affine_from_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert optimisation parameters into affine matrix and translation."""
    matrix = params[:9].reshape(3, 3)
    translation = params[9:12]
    return matrix, translation


def generalized_dice_affine_registration(
    fixed_seg,
    moving_seg,
    roi_mask,
    outprefix: Path,
    max_points: int = 25000,
):
    """Affine registration of labels by minimizing soft generalized Dice loss."""

    outprefix.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(f"{outprefix}0GenericAffine.mat")

    fixed_arr = fixed_seg.numpy().round().astype(np.uint8)
    moving_arr = moving_seg.numpy().round().astype(np.uint8)
    roi = roi_mask.numpy() > 0
    coords = np.argwhere(roi & np.isin(fixed_arr, [1, 2]))

    if coords.shape[0] < 50:
        center = _ants_indices_to_physical(
            fixed_seg, np.array([(np.asarray(fixed_arr.shape) - 1) / 2.0])
        )[0]
        transform = ants.create_ants_transform(
            transform_type="AffineTransform",
            dimension=3,
            matrix=np.eye(3).tolist(),
            translation=np.zeros(3),
            center=center,
        )
        ants.write_transform(transform, str(out_path))
        return {"fwdtransforms": [str(out_path)]}

    if coords.shape[0] > max_points:
        rng = np.random.default_rng(0)
        coords = coords[rng.choice(coords.shape[0], size=max_points, replace=False)]

    fixed_points = _ants_indices_to_physical(fixed_seg, coords.astype(float))
    center = fixed_points.mean(axis=0)
    fixed_labels = fixed_arr[coords[:, 0], coords[:, 1], coords[:, 2]]
    fixed_one_hot = np.stack(
        [(fixed_labels == 1).astype(float), (fixed_labels == 2).astype(float)], axis=0
    )
    class_volumes = fixed_one_hot.sum(axis=1)
    weights = 1.0 / np.maximum(class_volumes, 1.0) ** 2
    moving_one_hot = [(moving_arr == label).astype(float) for label in (1, 2)]

    def loss(params: np.ndarray) -> float:
        """Evaluate the registration loss for one affine parameter vector."""
        matrix, translation = _affine_from_params(params)
        try:
            inv_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return 1.0

        moving_points = center + (fixed_points - center - translation) @ inv_matrix.T
        moving_idx = _ants_physical_to_indices(moving_seg, moving_points)
        sampled = []
        sample_coords = [moving_idx[:, axis] for axis in range(3)]
        for label_img in moving_one_hot:
            sampled.append(
                ndimage.map_coordinates(
                    label_img,
                    sample_coords,
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
            )
        moving_soft = np.stack(sampled, axis=0)
        intersection = (moving_soft * fixed_one_hot).sum(axis=1)
        denominator = (moving_soft + fixed_one_hot).sum(axis=1)
        dice = (
            2.0
            * np.sum(weights * intersection)
            / max(float(np.sum(weights * denominator)), 1e-8)
        )
        return float(1.0 - dice)

    initial = np.concatenate([np.eye(3).ravel(), np.zeros(3)])
    bounds = []
    for row in range(3):
        for col in range(3):
            if row == col:
                bounds.append((0.85, 1.15))
            else:
                bounds.append((-0.15, 0.15))
    bounds.extend([(-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0)])
    result = optimize.minimize(
        loss,
        initial,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 60, "xtol": 1e-3, "ftol": 1e-4, "disp": False},
    )
    matrix, translation = _affine_from_params(result.x)
    transform = ants.create_ants_transform(
        transform_type="AffineTransform",
        dimension=3,
        matrix=matrix.tolist(),
        translation=translation,
        center=center,
    )
    ants.write_transform(transform, str(out_path))
    return {"fwdtransforms": [str(out_path)]}


def safe_registration(
    fixed, moving, metric: str, outprefix: Path, is_seg: bool = False
):
    """Run ANTs affine registration with a consistent output prefix."""
    outprefix.parent.mkdir(parents=True, exist_ok=True)
    if is_seg:
        return ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine",
            aff_metric="meansquares",
            outprefix=str(outprefix),
            initial_transform="Identity",
            verbose=False,
        )
    return ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Affine",
        aff_metric=metric,
        outprefix=str(outprefix),
        initial_transform="Identity",
        verbose=False,
    )
