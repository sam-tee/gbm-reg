"""
2-D slice-to-slice registration for preprocessed GBM MRI data.

For each post-mortem slab, the middle slice of the slab T2 volume is extracted
along the slab's thin axis. It is then registered in-plane to every
corresponding pre-mortem full-brain slice that lies within the tumour ROI along
that same axis. Results are ranked by mutual information for intensity mode, or
Dice overlap for label mode.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import ants
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.classes import Dirs
from src.misc import load_toml
from src.registration.grid_register import _bbox, _reorient

TOP_N = 5
RAS = "RAS"


def _mask_np(img: ants.ANTsImage) -> np.ndarray:
    """Build or transform binary mask data used by the workflow.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    data = img.numpy()
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(img.shape, dtype=bool)
    threshold = max(float(np.percentile(np.abs(data[finite]), 1)), 1e-6)
    return finite & (np.abs(data) > threshold)


def _thin_axis(img: ants.ANTsImage) -> int:
    """Return the axis with the smallest image extent.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
    
    Returns:
        int: Integer index, count, or label value.
        """
    return int(np.argmin(img.shape))


def _slice_np(img: ants.ANTsImage, axis: int, index: int) -> np.ndarray:
    """Extract one 2D NumPy slice from an ANTs image.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
        axis (int): Axis index along which to operate.
        index (int): Voxel or candidate index used by the operation.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    index = int(np.clip(index, 0, img.shape[axis] - 1))
    return np.take(img.numpy(), index, axis=axis).astype(np.float32)


def _slice_spacing(img: ants.ANTsImage, axis: int) -> tuple[float, float]:
    """Return the in-plane spacing for a slice axis.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
        axis (int): Axis index along which to operate.
    
    Returns:
        tuple[float, float]: Tuple containing the values described by the return annotation.
        """
    return tuple(float(s) for i, s in enumerate(img.spacing) if i != axis)


def _as_2d_image(data: np.ndarray, spacing: tuple[float, float]) -> ants.ANTsImage:
    """Wrap a 2D array as an ANTs image with slice spacing.
    
    Args:
        data (np.ndarray): Array data to process or display.
        spacing (tuple[float, float]): Voxel spacing associated with image data.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    return ants.from_numpy(data.astype(np.float32), spacing=spacing)


def _candidate_indices(tumour_labels: ants.ANTsImage, axis: int) -> list[int]:
    """Compute a Dice overlap score between label images.
    
    Args:
        tumour_labels (ants.ANTsImage): Tumour label image used to localise the tumour region.
        axis (int): Axis index along which to operate.
    
    Returns:
        list[int]: List containing the generated or resolved values.
        """
    tumour_min, tumour_max = _bbox(tumour_labels)
    lo = int(tumour_min[axis])
    hi = int(tumour_max[axis])
    return list(range(lo, hi + 1))


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


def _score(fixed: ants.ANTsImage, moving: ants.ANTsImage, mode: str) -> float:
    """Score a registration result using Dice or mutual information.
    
    Args:
        fixed (ants.ANTsImage): Fixed/reference image for comparison or registration.
        moving (ants.ANTsImage): Moving image to transform or compare against the fixed image.
        mode (str): Processing mode controlling which images, labels, or metric are used.
    
    Returns:
        float: Floating-point metric or coordinate value.
        """
    if mode == "label":
        return _dice(fixed, moving)
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


def _display(data: np.ndarray) -> np.ndarray:
    """Rotate slice data into display orientation.
    
    Args:
        data (np.ndarray): Array data to process or display.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    return np.rot90(data)


def _window(data: np.ndarray) -> tuple[float, float]:
    """Compute robust display intensity limits for image data.
    
    Args:
        data (np.ndarray): Array data to process or display.
    
    Returns:
        tuple[float, float]: Tuple containing the values described by the return annotation.
        """
    values = data[np.isfinite(data)]
    values = values[np.abs(values) > 1e-6]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [1, 99])
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def _write_comparison_png(
    fixed_slice: ants.ANTsImage,
    warped_slice: ants.ANTsImage,
    out_path: Path,
    title: str,
    mode: str,
) -> None:
    """Write a side-by-side fixed and warped slice comparison image.
    
    Args:
        fixed_slice (ants.ANTsImage): Slice data or slice identifier used by the operation.
        warped_slice (ants.ANTsImage): Slice data or slice identifier used by the operation.
        out_path (Path): Destination path for the generated output.
        title (str): Title to display on the generated plot.
        mode (str): Processing mode controlling which images, labels, or metric are used.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    fixed_np = fixed_slice.numpy()
    warped_np = warped_slice.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)

    vmin, vmax = _window(fixed_np)
    axes[0].imshow(_display(fixed_np), cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("full brain slice", fontsize=10)
    axes[0].axis("off")

    if mode == "label":
        axes[1].imshow(_display(warped_np), cmap="tab20", interpolation="nearest")
    else:
        vmin, vmax = _window(warped_np)
        axes[1].imshow(_display(warped_np), cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("registered slab slice", fontsize=10)
    axes[1].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _register_slice(
    fixed_2d: ants.ANTsImage,
    moving_2d: ants.ANTsImage,
    mode: str,
) -> dict | None:
    """Run one registration attempt and collect its score and outputs.
    
    Args:
        fixed_2d (ants.ANTsImage): Fixed 2d value used by the operation.
        moving_2d (ants.ANTsImage): Moving 2d value used by the operation.
        mode (str): Processing mode controlling which images, labels, or metric are used.
    
    Returns:
        dict | None: Dictionary containing generated metadata, paths, scores, or configuration values.
        """
    try:
        reg = ants.registration(
            fixed=fixed_2d,
            moving=moving_2d,
            type_of_transform="Rigid",
            aff_metric="meansquares" if mode == "label" else "mattes",
            reg_iterations=(100, 50, 0),
        )
    except RuntimeError as exc:
        print(f"      registration failed: {str(exc)[:120]}")
        return None

    transforms = reg.get("fwdtransforms", [])
    if not transforms:
        return None
    warped = ants.apply_transforms(
        fixed=fixed_2d,
        moving=moving_2d,
        transformlist=transforms,
        interpolator="nearestNeighbor" if mode == "label" else "linear",
    )
    if np.count_nonzero(np.abs(warped.numpy()) > 1e-6) == 0:
        return None
    return {
        "warped": warped,
        "transforms": transforms,
        "score": _score(fixed_2d, warped, mode),
    }


def _mode_paths(pre_dir: Path, slab_dir: Path, mode: str) -> tuple[Path, Path] | None:
    """Resolve fixed and moving image paths for a registration mode.
    
    Args:
        pre_dir (Path): Directory containing pre-mortem/preprocessed reference data.
        slab_dir (Path): Directory containing one post-mortem slab series.
        mode (str): Processing mode controlling which images, labels, or metric are used.
    
    Returns:
        tuple[Path, Path] | None: Tuple containing the values described by the return annotation.
        """
    if mode == "intensity":
        pre_path = pre_dir / "t2.nii.gz"
        slab_path = slab_dir / "t2.nii.gz"
    else:
        pre_path = pre_dir / "tissue_labels.nii.gz"
        slab_path = slab_dir / "segment_t2.nii.gz"
    if not pre_path.exists() or not slab_path.exists():
        return None
    return pre_path, slab_path


def _save_top(results: list[dict], out_dir: Path, mode: str, top_n: int) -> None:
    """Save the best-scoring results and their manifest.
    
    Args:
        results (list[dict]): Registration result dictionaries to sort and save.
        out_dir (Path): Directory path used by the operation.
        mode (str): Processing mode controlling which images, labels, or metric are used.
        top_n (int): Maximum number of top-scoring results to retain.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    out_dir.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: r["score"], reverse=(mode == "label"))
    manifest = []
    for rank, result in enumerate(results[:top_n], start=1):
        prefix = f"rank_{rank:02d}"
        warped_path = out_dir / f"{prefix}_warped_slice.nii.gz"
        comparison_path = out_dir / f"{prefix}_comparison.png"
        ants.image_write(result["warped"], str(warped_path))
        _write_comparison_png(
            fixed_slice=result["fixed_slice"],
            warped_slice=result["warped"],
            out_path=comparison_path,
            title=(
                f"{result['patient']} {result['slab']} rank {rank} | "
                f"fixed slice {result['fixed_slice_index']} | "
                f"{'Dice' if mode == 'label' else 'MI'}={result['score']:.4g}"
            ),
            mode=mode,
        )
        transforms = _copy_transforms(result["transforms"], out_dir, prefix)
        manifest.append(
            {
                "rank": rank,
                "patient": result["patient"],
                "slab": result["slab"],
                "mode": mode,
                "score_name": "dice_overlap"
                if mode == "label"
                else "mutual_information",
                "score": float(result["score"]),
                "fixed_full_brain_scan": result["fixed_full_brain_scan"],
                "fixed_full_brain_scan_path": str(result["fixed_full_brain_scan_path"]),
                "slab_axis": result["slab_axis"],
                "slab_middle_index": result["slab_middle_index"],
                "fixed_axis": result["fixed_axis"],
                "fixed_slice_index": result["fixed_slice_index"],
                "warped_slice": str(warped_path),
                "comparison_png": str(comparison_path),
                "transforms": transforms,
            }
        )
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def run_patient_slab(
    patient: str,
    pre_dir: Path,
    slab_dir: Path,
    output_root: Path,
    mode: str,
    top_n: int,
) -> None:
    """Run slice-to-slice registration for one patient slab.
    
    Args:
        patient (str): Patient identifier associated with outputs.
        pre_dir (Path): Directory containing pre-mortem/preprocessed reference data.
        slab_dir (Path): Directory containing one post-mortem slab series.
        output_root (Path): Root directory where generated outputs are written.
        mode (str): Processing mode controlling which images, labels, or metric are used.
        top_n (int): Maximum number of top-scoring results to retain.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    paths = _mode_paths(pre_dir, slab_dir, mode)
    out_dir = output_root / patient / slab_dir.name / mode
    tumour_path = pre_dir / "tumour_labels.nii.gz"
    if paths is None or not tumour_path.exists():
        print(f"  [{patient}/{slab_dir.name}] missing files for {mode}, skipping")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "manifest.json", "w") as f:
            json.dump([], f)
        return

    fixed_vol = _reorient(ants.image_read(str(paths[0])))
    moving_vol = _reorient(ants.image_read(str(paths[1])))
    tumour_labels = _reorient(ants.image_read(str(tumour_path)))

    axis = _thin_axis(moving_vol)
    moving_index = moving_vol.shape[axis] // 2
    moving_slice = _as_2d_image(
        _slice_np(moving_vol, axis, moving_index),
        _slice_spacing(moving_vol, axis),
    )

    fixed_spacing = _slice_spacing(fixed_vol, axis)
    results = []
    for fixed_index in _candidate_indices(tumour_labels, axis):
        fixed_slice = _as_2d_image(
            _slice_np(fixed_vol, axis, fixed_index),
            fixed_spacing,
        )
        if np.count_nonzero(np.abs(fixed_slice.numpy()) > 1e-6) == 0:
            continue
        print(f"    {mode}: slab mid {moving_index} -> fixed slice {fixed_index}")
        result = _register_slice(fixed_slice, moving_slice, mode)
        if result is None:
            continue
        result.update(
            {
                "patient": patient,
                "slab": slab_dir.name,
                "fixed_slice": fixed_slice,
                "fixed_full_brain_scan": paths[0].name.removesuffix(".nii.gz"),
                "fixed_full_brain_scan_path": paths[0],
                "slab_axis": axis,
                "slab_middle_index": moving_index,
                "fixed_axis": axis,
                "fixed_slice_index": fixed_index,
            }
        )
        results.append(result)
        print(f"      score={result['score']:.6f}")

    if results:
        _save_top(results, out_dir, mode, top_n)
        print(
            f"  [{patient}/{slab_dir.name}] wrote top {min(top_n, len(results))} to {out_dir}"
        )
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "manifest.json", "w") as f:
            json.dump([], f)
        print(f"  [{patient}/{slab_dir.name}] no successful {mode} registrations")


def run_slice_to_slice(
    config_file: Path,
    output_dir: Path | None = None,
    preprocessed_dir: Path | None = None,
    patients: list[str] | None = None,
    mode: str = "intensity",
    top_n: int = TOP_N,
) -> None:
    """Run slice-to-slice registration over selected patients and slabs.
    
    Args:
        config_file (Path): Path to the TOML configuration file.
        output_dir (Path | None): Directory where generated outputs are written.
        preprocessed_dir (Path | None): Directory containing preprocessed patient data.
        patients (list[str] | None): Optional patient identifiers to process.
        mode (str): Processing mode controlling which images, labels, or metric are used.
        top_n (int): Maximum number of top-scoring results to retain.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    cfg = load_toml(config_file)
    dirs = Dirs(**cfg.get("dirs", {}))
    preprocessed = preprocessed_dir or dirs.preprocessed
    output_root = output_dir or (
        Path(cfg.get("dirs", {}).get("registered", "outputs")) / "slice_to_slice"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = (
        [preprocessed / p for p in patients]
        if patients
        else sorted(preprocessed.iterdir())
    )
    for patient_dir in patient_dirs:
        pre_dir = patient_dir / "PreMortem"
        post_dir = patient_dir / "PostMortem"
        if not pre_dir.exists() or not post_dir.exists():
            continue
        patient = patient_dir.name
        print(f"\nPatient {patient}")
        for slab_dir in sorted(p for p in post_dir.iterdir() if p.is_dir()):
            print(f"  Slab {slab_dir.name}")
            modes = ("intensity", "label") if mode == "both" else (mode,)
            for selected_mode in modes:
                run_patient_slab(
                    patient, pre_dir, slab_dir, output_root, selected_mode, top_n
                )


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for this script.
    
    Returns:
        argparse.Namespace: Result produced by the operation in the form described by the return annotation.
        """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--preprocessed-dir", type=Path, default=None)
    parser.add_argument("--patient", action="append", default=None)
    parser.add_argument(
        "--mode", choices=("intensity", "label", "both"), default="intensity"
    )
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_slice_to_slice(
        config_file=args.config,
        output_dir=args.output_dir,
        preprocessed_dir=args.preprocessed_dir,
        patients=args.patient,
        mode=args.mode,
        top_n=args.top_n,
    )
