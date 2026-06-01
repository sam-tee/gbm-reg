"""
Grid-initialised slab-to-volume registration for preprocessed GBM MRI data.

For each patient, each post-mortem slab, and each available post-mortem scan,
the moving slab is initialised at several positions along the A-P axis of the
pre-mortem tumour hemisphere. Each initialisation is refined with ANTs rigid
registration and scored with either mutual information (intensity mode) or Dice
overlap (label mode). The top ranked warped volumes, transforms, starting
positions, and scores are written to an output directory.
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
INTENSITY_SCANS = ("t1", "t2")
LABEL_SCANS = ("segment_t1", "segment_t2")
LEFT_CEREBRUM_LABELS = (2, 3, 10, 11, 12, 13, 17, 18, 26, 28)
RIGHT_CEREBRUM_LABELS = (41, 42, 49, 50, 51, 52, 53, 54, 58, 60)


@dataclass(frozen=True)
class GridStart:
    """Candidate grid-search start position for slab registration.
    
    Attributes:
        index (int): Configuration or state value used by the pipeline.
        center_index (tuple[float, float, float]): Configuration or state value used by the pipeline.
        center_physical (tuple[float, float, float]): Configuration or state value used by the pipeline.
        y_index (float): Configuration or state value used by the pipeline.
        y_physical (float): Configuration or state value used by the pipeline.
        """
    index: int
    center_index: tuple[float, float, float]
    center_physical: tuple[float, float, float]
    y_index: float
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
    coords = np.argwhere(np.abs(data) > 1e-6)
    if coords.size == 0:
        coords = np.asarray([np.asarray(img.shape, dtype=float) / 2.0])
    return _physical_point(img, coords.mean(axis=0))


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

    left_dx = abs(tumour_center[0] - left_center[0])
    right_dx = abs(tumour_center[0] - right_center[0])
    return "left" if left_dx < right_dx else "right"


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


def _masked_roi(
    img: ants.ANTsImage,
    tumour_labels: ants.ANTsImage,
    freesurfer: ants.ANTsImage,
) -> ants.ANTsImage:
    """Build or transform binary mask data used by the workflow.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
        tumour_labels (ants.ANTsImage): Tumour label image used to localise the tumour region.
        freesurfer (ants.ANTsImage): FreeSurfer/SynthSeg label image used to identify anatomy.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    return ants.mask_image(
        img, _cerebrum_mask(freesurfer, _hemisphere(tumour_labels, freesurfer))
    )


def _grid_starts(
    template: ants.ANTsImage,
    tumour_labels: ants.ANTsImage,
    freesurfer: ants.ANTsImage,
    num_starts: int,
) -> list[GridStart]:
    """Generate grid-search starting positions across the tumour-side cerebrum.
    
    Args:
        template (ants.ANTsImage): Template value used by the operation.
        tumour_labels (ants.ANTsImage): Tumour label image used to localise the tumour region.
        freesurfer (ants.ANTsImage): FreeSurfer/SynthSeg label image used to identify anatomy.
        num_starts (int): Num starts value used by the operation.
    
    Returns:
        list[GridStart]: List containing the generated or resolved values.
        """
    tumour_min, tumour_max = _bbox(tumour_labels)
    hemi_mask = _cerebrum_mask(freesurfer, _hemisphere(tumour_labels, freesurfer))
    hemi_min, hemi_max = _bbox(hemi_mask)

    x_center = (hemi_min[0] + hemi_max[0]) / 2.0
    z_center = (tumour_min[2] + tumour_max[2]) / 2.0
    y_positions = np.linspace(tumour_min[1], tumour_max[1], max(num_starts, 1))

    starts = []
    for i, y in enumerate(y_positions):
        idx = (float(x_center), float(y), float(z_center))
        phys = _physical_point(template, idx)
        starts.append(
            GridStart(
                index=i,
                center_index=idx,
                center_physical=tuple(float(v) for v in phys),
                y_index=float(y),
                y_physical=float(phys[1]),
            )
        )
    return starts


def _slice_around_y(
    img: ants.ANTsImage,
    center_y: float,
    thickness_mm: float,
) -> ants.ANTsImage:
    """Extract a slab-thickness coronal slice around a candidate y position.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
        center_y (float): Candidate centre location along the y-axis.
        thickness_mm (float): Thickness mm value used by the operation.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    half = max(int(round((thickness_mm / img.spacing[1]) / 2.0)), 1)
    lo_y = max(int(round(center_y)) - half, 0)
    hi_y = min(int(round(center_y)) + half + 1, img.shape[1])
    return ants.crop_indices(img, [0, lo_y, 0], [img.shape[0], hi_y, img.shape[2]])


def _write_initial_transform(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    start: GridStart,
) -> str:
    """Write an Euler transform that initialises moving-to-fixed alignment.
    
    Args:
        fixed (ants.ANTsImage): Fixed/reference image for comparison or registration.
        moving (ants.ANTsImage): Moving image to transform or compare against the fixed image.
        start (GridStart): Start value used by the operation.
    
    Returns:
        str: String path, label, mode, or identifier produced by the operation.
        """
    moving_center = _center_of_mass_physical(moving)
    # ANTs applies this Euler transform in the fixed-to-moving direction when
    # resampling, so the stored translation is the inverse of the desired
    # moving-centre-to-start displacement.
    translation = moving_center - np.asarray(start.center_physical)
    tx = ants.create_ants_transform(
        transform_type="Euler3DTransform",
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


def _dice(fixed: ants.ANTsImage, moving: ants.ANTsImage) -> float:
    """Compute a Dice overlap score between label images.
    
    Args:
        fixed (ants.ANTsImage): Fixed/reference image for comparison or registration.
        moving (ants.ANTsImage): Moving image to transform or compare against the fixed image.
    
    Returns:
        float: Floating-point metric or coordinate value.
        """
    fixed_np = fixed.numpy() > 0
    moving_np = moving.numpy() > 0
    denom = int(fixed_np.sum() + moving_np.sum())
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(fixed_np, moving_np).sum() / denom)


def _register_one(
    fixed_slice: ants.ANTsImage,
    fixed_full: ants.ANTsImage,
    moving: ants.ANTsImage,
    start: GridStart,
    mode: str,
) -> dict | None:
    """Run one registration attempt and collect its score and outputs.
    
    Args:
        fixed_slice (ants.ANTsImage): Slice data or slice identifier used by the operation.
        fixed_full (ants.ANTsImage): Fixed full value used by the operation.
        moving (ants.ANTsImage): Moving image to transform or compare against the fixed image.
        start (GridStart): Start value used by the operation.
        mode (str): Processing mode controlling which images, labels, or metric are used.
    
    Returns:
        dict | None: Dictionary containing generated metadata, paths, scores, or configuration values.
        """
    init_tx = _write_initial_transform(fixed_slice, moving, start)
    try:
        kwargs = {
            "fixed": fixed_slice,
            "moving": moving,
            "type_of_transform": "Rigid",
            "initial_transform": [init_tx],
            "reg_iterations": (80, 40, 0),
        }
        if mode == "intensity":
            kwargs.update({"aff_metric": "mattes", "syn_metric": "mattes"})
        else:
            kwargs.update({"aff_metric": "meansquares", "syn_metric": "meansquares"})

        reg = ants.registration(**kwargs)
        transformlist = reg.get("fwdtransforms", [])
        if not transformlist:
            return None

        interpolator = "linear" if mode == "intensity" else "nearestNeighbor"
        warped_slice = ants.apply_transforms(
            fixed=fixed_slice,
            moving=moving,
            transformlist=transformlist,
            interpolator=interpolator,
        )
        if np.count_nonzero(np.abs(warped_slice.numpy()) > 1e-6) == 0:
            return None

        score = (
            _mutual_information(fixed_slice, warped_slice)
            if mode == "intensity"
            else _dice(fixed_slice, warped_slice)
        )
        warped_full = ants.apply_transforms(
            fixed=fixed_full,
            moving=moving,
            transformlist=transformlist,
            interpolator=interpolator,
        )
        return {
            "score": score,
            "warped_full": warped_full,
            "transformlist": transformlist,
        }
    except RuntimeError as exc:
        print(f"      registration failed: {str(exc)[:120]}")
        return None
    finally:
        Path(init_tx).unlink(missing_ok=True)


def _copy_transforms(transformlist: list[str], out_dir: Path, prefix: str) -> list[str]:
    """Copy generated transform files into the result directory.
    
    Args:
        transformlist (list[str]): Transform file paths produced by registration.
        out_dir (Path): Directory path used by the operation.
        prefix (str): Filename prefix for copied or generated outputs.
    
    Returns:
        list[str]: List containing the generated or resolved values.
        """
    paths = []
    for i, tx in enumerate(transformlist):
        src = Path(tx)
        dst = out_dir / f"{prefix}_transform_{i:02d}{src.suffix}"
        shutil.copy2(src, dst)
        paths.append(str(dst))
    return paths


def _save_top_results(
    results: list[dict], out_dir: Path, top_n: int, mode: str
) -> None:
    """Save the best-scoring registration results and their manifest.
    
    Args:
        results (list[dict]): Registration result dictionaries to sort and save.
        out_dir (Path): Directory path used by the operation.
        top_n (int): Maximum number of top-scoring results to retain.
        mode (str): Processing mode controlling which images, labels, or metric are used.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    reverse = mode == "label"
    results.sort(key=lambda r: r["score"], reverse=reverse)
    top = results[:top_n]
    manifest = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, result in enumerate(top, start=1):
        prefix = f"rank_{rank:02d}"
        volume_path = out_dir / f"{prefix}_warped.nii.gz"
        ants.image_write(result["warped_full"], str(volume_path))
        transform_paths = _copy_transforms(result["transformlist"], out_dir, prefix)
        start = result["start"]
        manifest.append(
            {
                "rank": rank,
                "patient": result["patient"],
                "slab": result["slab"],
                "scan": result["scan"],
                "mode": mode,
                "score_name": "mutual_information"
                if mode == "intensity"
                else "dice_overlap",
                "score": float(result["score"]),
                "start_index": start.index,
                "start_center_index": list(start.center_index),
                "start_center_physical": list(start.center_physical),
                "start_y_index": start.y_index,
                "start_y_physical": start.y_physical,
                "warped_volume": str(volume_path),
                "transforms": transform_paths,
            }
        )

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _scan_pair(
    pre_dir: Path, slab_dir: Path, scan: str, mode: str
) -> tuple[Path, Path] | None:
    """Evaluate candidate alignments for one fixed/moving image pair.
    
    Args:
        pre_dir (Path): Directory containing pre-mortem/preprocessed reference data.
        slab_dir (Path): Directory containing one post-mortem slab series.
        scan (str): Scan name or scan phase to process.
        mode (str): Processing mode controlling which images, labels, or metric are used.
    
    Returns:
        tuple[Path, Path] | None: Tuple containing the values described by the return annotation.
        """
    if mode == "intensity":
        pre_path = pre_dir / f"{scan}.nii.gz"
        post_path = slab_dir / f"{scan}.nii.gz"
    else:
        pre_path = pre_dir / "tissue_labels.nii.gz"
        post_path = slab_dir / f"{scan}.nii.gz"
    if not pre_path.exists() or not post_path.exists():
        return None
    return pre_path, post_path


def _run_scan(
    patient: str,
    pre_dir: Path,
    slab_dir: Path,
    scan: str,
    mode: str,
    output_root: Path,
    starts: list[GridStart],
    tumour_labels: ants.ANTsImage,
    freesurfer: ants.ANTsImage,
    top_n: int,
    slab_thickness_mm: float,
) -> None:
    """Run the grid-registration scan across patients and slabs.
    
    Args:
        patient (str): Patient identifier associated with outputs.
        pre_dir (Path): Directory containing pre-mortem/preprocessed reference data.
        slab_dir (Path): Directory containing one post-mortem slab series.
        scan (str): Scan name or scan phase to process.
        mode (str): Processing mode controlling which images, labels, or metric are used.
        output_root (Path): Root directory where generated outputs are written.
        starts (list[GridStart]): Starts value used by the operation.
        tumour_labels (ants.ANTsImage): Tumour label image used to localise the tumour region.
        freesurfer (ants.ANTsImage): FreeSurfer/SynthSeg label image used to identify anatomy.
        top_n (int): Maximum number of top-scoring results to retain.
        slab_thickness_mm (float): Slab thickness mm value used by the operation.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    pair = _scan_pair(pre_dir, slab_dir, scan, mode)
    if pair is None:
        print(f"    skipping missing {mode}/{scan}")
        return

    fixed_full = _reorient(ants.image_read(str(pair[0])))
    moving = _reorient(ants.image_read(str(pair[1])))
    if mode == "intensity":
        fixed_full = _masked_roi(fixed_full, tumour_labels, freesurfer)
        moving = ants.mask_image(
            moving, _reorient(ants.image_read(str(slab_dir / "brain_mask.nii.gz")))
        )
    else:
        fixed_full = _masked_roi(fixed_full, tumour_labels, freesurfer)

    results = []
    for start in starts:
        fixed_slice = _slice_around_y(fixed_full, start.y_index, slab_thickness_mm)
        print(f"      start {start.index:02d} y={start.y_index:.1f}")
        result = _register_one(fixed_slice, fixed_full, moving, start, mode)
        if result is None:
            continue
        result.update(
            {
                "patient": patient,
                "slab": slab_dir.name,
                "scan": scan,
                "start": start,
            }
        )
        results.append(result)
        print(f"        score={result['score']:.6f}")

    out_dir = output_root / patient / slab_dir.name / mode / scan
    if results:
        _save_top_results(results, out_dir, top_n, mode)
        print(f"    wrote {min(len(results), top_n)} ranked result(s) to {out_dir}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "manifest.json", "w") as f:
            json.dump([], f)
        print(f"    no successful registrations for {mode}/{scan}")


def run_all_registrations(
    config_file: Path,
    output_dir: Path | None = None,
    patients: list[str] | None = None,
    modes: tuple[str, ...] = ("intensity", "label"),
    num_starts: int = 9,
    top_n: int = 5,
    slab_thickness_mm: float = 12.0,
) -> None:
    """Run all configured registration modes for selected patients.
    
    Args:
        config_file (Path): Path to the TOML configuration file.
        output_dir (Path | None): Directory where generated outputs are written.
        patients (list[str] | None): Optional patient identifiers to process.
        modes (tuple[str, ...]): Optional modes value. Defaults to `('intensity', 'label')`.
        num_starts (int): Optional num starts value. Defaults to `9`.
        top_n (int): Maximum number of top-scoring results to retain.
        slab_thickness_mm (float): Optional slab thickness mm value. Defaults to `12.0`.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    cfg = load_toml(config_file)
    dirs = Dirs(**cfg.get("dirs", {}))
    preprocessed = dirs.preprocessed
    output_root = output_dir or (
        Path(cfg.get("dirs", {}).get("registered", "outputs")) / "registration"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = (
        [preprocessed / p for p in patients]
        if patients
        else sorted(preprocessed.iterdir())
    )
    for patient_dir in patient_dirs:
        if not patient_dir.is_dir():
            continue
        patient = patient_dir.name
        pre_dir = patient_dir / "PreMortem"
        post_dir = patient_dir / "PostMortem"
        if not pre_dir.exists() or not post_dir.exists():
            continue

        print(f"\nPatient {patient}")
        tumour_labels = _reorient(
            ants.image_read(str(pre_dir / "tumour_labels.nii.gz"))
        )
        freesurfer = _reorient(
            ants.image_read(str(pre_dir / "freesurfer_labels.nii.gz"))
        )
        template = _reorient(ants.image_read(str(pre_dir / "t1.nii.gz")))
        starts = _grid_starts(template, tumour_labels, freesurfer, num_starts)

        for slab_dir in sorted(p for p in post_dir.iterdir() if p.is_dir()):
            print(f"  Slab {slab_dir.name}")
            if "intensity" in modes:
                for scan in INTENSITY_SCANS:
                    print(f"    intensity/{scan}")
                    _run_scan(
                        patient,
                        pre_dir,
                        slab_dir,
                        scan,
                        "intensity",
                        output_root,
                        starts,
                        tumour_labels,
                        freesurfer,
                        top_n,
                        slab_thickness_mm,
                    )
            if "label" in modes:
                for scan in LABEL_SCANS:
                    print(f"    label/{scan}")
                    _run_scan(
                        patient,
                        pre_dir,
                        slab_dir,
                        scan,
                        "label",
                        output_root,
                        starts,
                        tumour_labels,
                        freesurfer,
                        top_n,
                        slab_thickness_mm,
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
    parser.add_argument(
        "--mode",
        choices=("intensity", "label", "both"),
        default="both",
        help="Registration mode to run.",
    )
    parser.add_argument("--num-starts", type=int, default=9)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--slab-thickness-mm", type=float, default=12.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    selected_modes = ("intensity", "label") if args.mode == "both" else (args.mode,)
    run_all_registrations(
        config_file=args.config,
        output_dir=args.output_dir,
        patients=args.patient,
        modes=selected_modes,
        num_starts=args.num_starts,
        top_n=args.top_n,
        slab_thickness_mm=args.slab_thickness_mm,
    )
