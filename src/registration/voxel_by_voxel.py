"""
Voxel-by-voxel A-P slab-to-volume registration.

For each patient and each post-mortem series, this registers the post-mortem
T2 slab to the pre-mortem T1c tumour-side hemisphere. The moving slab is
initialised once for every voxel position along the anterior-posterior axis of
the pre-mortem tumour bounding box. At each position, the slab centre is placed
at the hemisphere centre in X/Z and at the current tumour A-P voxel in Y, then
refined with affine registration. The top five registrations are written under
``outputs/VbyV`` by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ants
import numpy as np

from src.classes import Dirs
from src.misc import load_toml

RAS = "RAS"
TOP_N = 5
LEFT_CEREBRUM_LABELS = (2, 3, 10, 11, 12, 13, 17, 18, 26, 28)
RIGHT_CEREBRUM_LABELS = (41, 42, 49, 50, 51, 52, 53, 54, 58, 60)


@dataclass(frozen=True)
class VoxelStart:
    """Candidate voxel-search start position for slab registration.
    
    Attributes:
        index (int): Configuration or state value used by the pipeline.
        center_index (tuple[float, float, float]): Configuration or state value used by the pipeline.
        center_physical (tuple[float, float, float]): Configuration or state value used by the pipeline.
        y_index (int): Configuration or state value used by the pipeline.
        y_physical (float): Configuration or state value used by the pipeline.
        """
    index: int
    center_index: tuple[float, float, float]
    center_physical: tuple[float, float, float]
    y_index: int
    y_physical: float


def _reorient(img: ants.ANTsImage) -> ants.ANTsImage:
    """Return the image reoriented into the canonical RAS orientation.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    return ants.reorient_image2(img, RAS)


def _bbox(img: ants.ANTsImage) -> tuple[np.ndarray, np.ndarray]:
    """Return the bounding box of foreground voxels.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the values described by the return annotation.
        """
    coords = np.argwhere(img.numpy() > 0)
    if coords.size == 0:
        raise ValueError("Cannot compute a bounding box for an empty image")
    return coords.min(axis=0), coords.max(axis=0)


def _physical_point(img: ants.ANTsImage, index: Iterable[float]) -> np.ndarray:
    """Convert a voxel index into a physical-space coordinate.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
        index (Iterable[float]): Voxel or candidate index used by the operation.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    idx = np.asarray(tuple(index), dtype=float)
    return np.asarray(img.origin) + np.asarray(img.direction).dot(idx * img.spacing)


def _center_of_mass_physical(img: ants.ANTsImage) -> np.ndarray:
    """Compute the physical-space centre of mass for non-zero voxels.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    data = img.numpy()
    coords = np.argwhere(np.isfinite(data) & (np.abs(data) > 1e-6))
    if coords.size == 0:
        coords = np.asarray([np.asarray(img.shape, dtype=float) / 2.0])
    return _physical_point(img, coords.mean(axis=0))


def _mask_center_index(mask: ants.ANTsImage) -> np.ndarray:
    """Build or transform binary mask data used by the workflow.
    
    Args:
        mask (ants.ANTsImage): Binary or label mask used to limit the operation.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    coords = np.argwhere(mask.numpy() > 0)
    if coords.size == 0:
        raise ValueError("Cannot compute centre for an empty mask")
    return coords.mean(axis=0)


def _mask_center_physical(reference: ants.ANTsImage, mask: np.ndarray) -> np.ndarray:
    """Build or transform binary mask data used by the workflow.
    
    Args:
        reference (ants.ANTsImage): Reference value used by the operation.
        mask (np.ndarray): Binary or label mask used to limit the operation.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot compute centre for an empty mask")
    return _physical_point(reference, coords.mean(axis=0))


def _hemisphere(tumour_labels: ants.ANTsImage, freesurfer: ants.ANTsImage) -> str:
    """Infer whether the tumour lies in the left or right cerebral hemisphere.
    
    Args:
        tumour_labels (ants.ANTsImage): Tumour label image used to localise the tumour region.
        freesurfer (ants.ANTsImage): FreeSurfer/SynthSeg label image used to identify anatomy.
    
    Returns:
        str: String path, label, mode, or identifier produced by the operation.
        """
    fs = freesurfer.numpy()
    tumour_center = _mask_center_physical(tumour_labels, tumour_labels.numpy() > 0)
    left_center = _mask_center_physical(freesurfer, np.isin(fs, LEFT_CEREBRUM_LABELS))
    right_center = _mask_center_physical(freesurfer, np.isin(fs, RIGHT_CEREBRUM_LABELS))
    return (
        "left"
        if abs(tumour_center[0] - left_center[0])
        < abs(tumour_center[0] - right_center[0])
        else "right"
    )


def _cerebrum_mask(freesurfer: ants.ANTsImage, hemisphere: str) -> ants.ANTsImage:
    """Build or transform binary mask data used by the workflow.
    
    Args:
        freesurfer (ants.ANTsImage): FreeSurfer/SynthSeg label image used to identify anatomy.
        hemisphere (str): Hemisphere value used by the operation.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    labels = LEFT_CEREBRUM_LABELS if hemisphere == "left" else RIGHT_CEREBRUM_LABELS
    mask = np.isin(freesurfer.numpy(), labels).astype(np.float32)
    return ants.new_image_like(freesurfer, mask)


def _masked_hemisphere(
    img: ants.ANTsImage,
    tumour_labels: ants.ANTsImage,
    freesurfer: ants.ANTsImage,
) -> tuple[ants.ANTsImage, ants.ANTsImage, str]:
    """Build or transform binary mask data used by the workflow.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
        tumour_labels (ants.ANTsImage): Tumour label image used to localise the tumour region.
        freesurfer (ants.ANTsImage): FreeSurfer/SynthSeg label image used to identify anatomy.
    
    Returns:
        tuple[ants.ANTsImage, ants.ANTsImage, str]: Tuple containing the values described by the return annotation.
        """
    side = _hemisphere(tumour_labels, freesurfer)
    mask = _cerebrum_mask(freesurfer, side)
    return ants.mask_image(img, mask), mask, side


def _voxel_starts(
    template: ants.ANTsImage,
    tumour_labels: ants.ANTsImage,
    hemisphere_mask: ants.ANTsImage,
) -> list[VoxelStart]:
    """Generate voxel-wise candidate starting positions for slab alignment.
    
    Args:
        template (ants.ANTsImage): Template value used by the operation.
        tumour_labels (ants.ANTsImage): Tumour label image used to localise the tumour region.
        hemisphere_mask (ants.ANTsImage): Mask data used to constrain or describe the operation.
    
    Returns:
        list[VoxelStart]: List containing the generated or resolved values.
        """
    tumour_min, tumour_max = _bbox(tumour_labels)
    hemi_center = _mask_center_index(hemisphere_mask)

    starts = []
    for index, y in enumerate(range(int(tumour_min[1]), int(tumour_max[1]) + 1)):
        center_index = (float(hemi_center[0]), float(y), float(hemi_center[2]))
        center_physical = _physical_point(template, center_index)
        starts.append(
            VoxelStart(
                index=index,
                center_index=center_index,
                center_physical=tuple(float(v) for v in center_physical),
                y_index=int(y),
                y_physical=float(center_physical[1]),
            )
        )
    return starts


def _slab_y_thickness_mm(slab: ants.ANTsImage) -> float:
    """Return the slab thickness along the y-axis in millimetres.
    
    Args:
        slab (ants.ANTsImage): Slab identifier or slab image to process.
    
    Returns:
        float: Floating-point metric or coordinate value.
        """
    return max(float(slab.shape[1] * slab.spacing[1]), float(slab.spacing[1]))


def _slice_around_y(
    img: ants.ANTsImage,
    center_y: int,
    thickness_mm: float,
) -> ants.ANTsImage:
    """Extract a slab-thickness coronal slice around a candidate y position.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
        center_y (int): Candidate centre location along the y-axis.
        thickness_mm (float): Thickness mm value used by the operation.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    half = max(int(round((thickness_mm / img.spacing[1]) / 2.0)), 1)
    lo_y = max(int(center_y) - half, 0)
    hi_y = min(int(center_y) + half + 1, img.shape[1])
    return ants.crop_indices(img, [0, lo_y, 0], [img.shape[0], hi_y, img.shape[2]])


def _write_initial_transform(moving: ants.ANTsImage, start: VoxelStart) -> str:
    """Write an Euler transform that initialises moving-to-fixed alignment.
    
    Args:
        moving (ants.ANTsImage): Moving image to transform or compare against the fixed image.
        start (VoxelStart): Start value used by the operation.
    
    Returns:
        str: String path, label, mode, or identifier produced by the operation.
        """
    moving_center = _center_of_mass_physical(moving)
    translation = moving_center - np.asarray(start.center_physical)
    tx = ants.create_ants_transform(
        transform_type="AffineTransform",
        dimension=3,
        translation=tuple(float(v) for v in translation),
    )
    f = tempfile.NamedTemporaryFile(suffix=".mat", delete=False)
    f.close()
    ants.write_transform(tx, f.name)
    return f.name


def _mutual_information(fixed: ants.ANTsImage, moving: ants.ANTsImage) -> float:
    """Compute mutual information between two image arrays.
    
    Args:
        fixed (ants.ANTsImage): Fixed/reference image for comparison or registration.
        moving (ants.ANTsImage): Moving image to transform or compare against the fixed image.
    
    Returns:
        float: Floating-point metric or coordinate value.
        """
    return float(
        ants.image_mutual_information(fixed.clone("float"), moving.clone("float"))
    )


def _copy_transforms(transformlist: list[str], out_dir: Path, prefix: str) -> list[str]:
    """Copy generated transform files into the result directory.
    
    Args:
        transformlist (list[str]): Transform file paths produced by registration.
        out_dir (Path): Directory path used by the operation.
        prefix (str): Filename prefix for copied or generated outputs.
    
    Returns:
        list[str]: List containing the generated or resolved values.
        """
    copied = []
    for i, tx in enumerate(transformlist):
        src = Path(tx)
        dst = out_dir / f"{prefix}_transform_{i:02d}{src.suffix}"
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def _register_one(
    fixed_slice: ants.ANTsImage,
    fixed_full: ants.ANTsImage,
    moving: ants.ANTsImage,
    start: VoxelStart,
) -> dict | None:
    """Run one registration attempt and collect its score and outputs.
    
    Args:
        fixed_slice (ants.ANTsImage): Slice data or slice identifier used by the operation.
        fixed_full (ants.ANTsImage): Fixed full value used by the operation.
        moving (ants.ANTsImage): Moving image to transform or compare against the fixed image.
        start (VoxelStart): Start value used by the operation.
    
    Returns:
        dict | None: Dictionary containing generated metadata, paths, scores, or configuration values.
        """
    init_tx = _write_initial_transform(moving, start)
    try:
        reg = ants.registration(
            fixed=fixed_slice,
            moving=moving,
            type_of_transform="Affine",
            initial_transform=[init_tx],
            aff_metric="mattes",
            syn_metric="mattes",
            reg_iterations=(100, 50, 0),
        )
        transforms = reg.get("fwdtransforms", [])
        if not transforms:
            return None

        warped_slice = ants.apply_transforms(
            fixed=fixed_slice,
            moving=moving,
            transformlist=transforms,
            interpolator="linear",
        )
        if np.count_nonzero(np.abs(warped_slice.numpy()) > 1e-6) == 0:
            return None

        warped_full = ants.apply_transforms(
            fixed=fixed_full,
            moving=moving,
            transformlist=transforms,
            interpolator="linear",
        )
        return {
            "score": _mutual_information(fixed_slice, warped_slice),
            "warped_full": warped_full,
            "warped_slice": warped_slice,
            "transformlist": transforms,
        }
    except RuntimeError as exc:
        print(f"      registration failed: {str(exc)[:120]}")
        return None
    finally:
        Path(init_tx).unlink(missing_ok=True)


def _save_top(results: list[dict], out_dir: Path, top_n: int) -> None:
    """Save the best-scoring results and their manifest.
    
    Args:
        results (list[dict]): Registration result dictionaries to sort and save.
        out_dir (Path): Directory path used by the operation.
        top_n (int): Maximum number of top-scoring results to retain.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    out_dir.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: r["score"])
    manifest = []

    for rank, result in enumerate(results[:top_n], start=1):
        prefix = f"rank_{rank:02d}"
        volume_path = out_dir / f"{prefix}_warped_volume.nii.gz"
        slice_path = out_dir / f"{prefix}_warped_scored_slab.nii.gz"
        ants.image_write(result["warped_full"], str(volume_path))
        ants.image_write(result["warped_slice"], str(slice_path))
        transforms = _copy_transforms(result["transformlist"], out_dir, prefix)
        start = result["start"]
        manifest.append(
            {
                "rank": rank,
                "patient": result["patient"],
                "series": result["series"],
                "fixed_scan": "t1c",
                "moving_scan": "t2",
                "score_name": "mutual_information",
                "score": float(result["score"]),
                "hemisphere": result["hemisphere"],
                "start_index": start.index,
                "start_center_index": list(start.center_index),
                "start_center_physical": list(start.center_physical),
                "start_y_index": start.y_index,
                "start_y_physical": start.y_physical,
                "warped_volume": str(volume_path),
                "warped_scored_slab": str(slice_path),
                "transforms": transforms,
            }
        )

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _run_series(
    patient: str,
    pre_dir: Path,
    series_dir: Path,
    output_root: Path,
    tumour_labels: ants.ANTsImage,
    freesurfer: ants.ANTsImage,
    top_n: int,
) -> None:
    """Run voxel-by-voxel registration for one slab series.
    
    Args:
        patient (str): Patient identifier associated with outputs.
        pre_dir (Path): Directory containing pre-mortem/preprocessed reference data.
        series_dir (Path): Directory path used by the operation.
        output_root (Path): Root directory where generated outputs are written.
        tumour_labels (ants.ANTsImage): Tumour label image used to localise the tumour region.
        freesurfer (ants.ANTsImage): FreeSurfer/SynthSeg label image used to identify anatomy.
        top_n (int): Maximum number of top-scoring results to retain.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    t1c_path = pre_dir / "t1c.nii.gz"
    t2_path = series_dir / "t2.nii.gz"
    if not t1c_path.exists() or not t2_path.exists():
        print(f"  [{patient}/{series_dir.name}] missing t1c or t2, skipping")
        return

    fixed_full = _reorient(ants.image_read(str(t1c_path)))
    moving = _reorient(ants.image_read(str(t2_path)))
    fixed_full, hemi_mask, hemi = _masked_hemisphere(
        fixed_full, tumour_labels, freesurfer
    )

    brain_mask_path = series_dir / "brain_mask.nii.gz"
    if brain_mask_path.exists():
        moving = ants.mask_image(
            moving, _reorient(ants.image_read(str(brain_mask_path)))
        )

    starts = _voxel_starts(fixed_full, tumour_labels, hemi_mask)
    thickness_mm = _slab_y_thickness_mm(moving)
    print(
        f"  Series {series_dir.name}: {len(starts)} A-P voxel starts, "
        f"hemisphere={hemi}, slab_y_thickness={thickness_mm:.2f} mm"
    )

    results = []
    for start in starts:
        fixed_slice = _slice_around_y(fixed_full, start.y_index, thickness_mm)
        print(f"    voxel {start.index:03d} y={start.y_index}")
        result = _register_one(fixed_slice, fixed_full, moving, start)
        if result is None:
            continue
        result.update(
            {
                "patient": patient,
                "series": series_dir.name,
                "hemisphere": hemi,
                "start": start,
            }
        )
        results.append(result)
        print(f"      MI={result['score']:.6f}")

    out_dir = output_root / patient / series_dir.name
    if results:
        _save_top(results, out_dir, top_n)
        print(
            f"  [{patient}/{series_dir.name}] wrote top {min(top_n, len(results))} to {out_dir}"
        )
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "manifest.json", "w") as f:
            json.dump([], f)
        print(f"  [{patient}/{series_dir.name}] no successful registrations")


def run_voxel_by_voxel(
    config_file: Path,
    output_dir: Path | None = None,
    patients: list[str] | None = None,
    top_n: int = TOP_N,
) -> None:
    """Run voxel-by-voxel registration over selected patients and slabs.
    
    Args:
        config_file (Path): Path to the TOML configuration file.
        output_dir (Path | None): Directory where generated outputs are written.
        patients (list[str] | None): Optional patient identifiers to process.
        top_n (int): Maximum number of top-scoring results to retain.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    cfg = load_toml(config_file)
    dirs = Dirs(**cfg.get("dirs", {}))
    output_root = output_dir or (
        Path(cfg.get("dirs", {}).get("registered", "outputs")) / "VbyV"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = (
        [dirs.preprocessed / p for p in patients]
        if patients
        else sorted(dirs.preprocessed.iterdir())
    )
    for patient_dir in patient_dirs:
        if not patient_dir.is_dir():
            continue
        pre_dir = patient_dir / "PreMortem"
        post_dir = patient_dir / "PostMortem"
        if not pre_dir.exists() or not post_dir.exists():
            continue

        patient = patient_dir.name
        print(f"\nPatient {patient}")
        tumour_labels = _reorient(
            ants.image_read(str(pre_dir / "tumour_labels.nii.gz"))
        )
        freesurfer = _reorient(
            ants.image_read(str(pre_dir / "freesurfer_labels.nii.gz"))
        )
        for series_dir in sorted(p for p in post_dir.iterdir() if p.is_dir()):
            _run_series(
                patient=patient,
                pre_dir=pre_dir,
                series_dir=series_dir,
                output_root=output_root,
                tumour_labels=tumour_labels,
                freesurfer=freesurfer,
                top_n=top_n,
            )


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for this script.
    
    Returns:
        argparse.Namespace: Result produced by the operation in the form described by the return annotation.
        """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--patient", action="append", default=None)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_voxel_by_voxel(
        config_file=args.config,
        output_dir=args.output_dir,
        patients=args.patient,
        top_n=args.top_n,
    )
