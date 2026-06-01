"""
Create visual summaries for grid registration outputs.

Each registration manifest gets a matching output folder containing:
  1. ``overview.png``: full-brain MRI views with top-5 slab contours;
  2. ``rank_XX_comparison.png``: full-brain MRI slice beside the corresponding
     warped slab slice for each ranked result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


COLORS = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange"]


def _load(path: Path) -> np.ndarray:
    """Load a NIfTI volume into a NumPy array.
    
    Args:
        path (Path): Filesystem path for the input or output resource.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    return np.asanyarray(nib.load(str(path)).dataobj)


def _take_slice(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Extract and orient one display slice from a volume.
    
    Args:
        vol (np.ndarray): Vol value used by the operation.
        axis (int): Axis index along which to operate.
        index (int): Voxel or candidate index used by the operation.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    index = int(np.clip(index, 0, vol.shape[axis] - 1))
    return np.rot90(np.take(vol, index, axis=axis))


def _mask(vol: np.ndarray) -> np.ndarray:
    """Build or transform binary mask data used by the workflow.
    
    Args:
        vol (np.ndarray): Vol value used by the operation.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    finite = np.isfinite(vol)
    if not finite.any():
        return np.zeros(vol.shape, dtype=bool)
    values = np.abs(vol[finite])
    threshold = max(float(np.percentile(values, 1)), 1e-6)
    return finite & (np.abs(vol) > threshold)


def _bbox(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the bounding box of foreground voxels.
    
    Args:
        mask (np.ndarray): Binary or label mask used to limit the operation.
    
    Returns:
        tuple[np.ndarray, np.ndarray] | None: Tuple containing the values described by the return annotation.
        """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.min(axis=0), coords.max(axis=0)


def _slice_axis_and_index(
    mask: np.ndarray, shape: tuple[int, int, int]
) -> tuple[int, int]:
    """Choose the best axis and index for a representative slice.
    
    Args:
        mask (np.ndarray): Binary or label mask used to limit the operation.
        shape (tuple[int, int, int]): Volume shape used for bounds checking.
    
    Returns:
        tuple[int, int]: Tuple containing the values described by the return annotation.
        """
    bbox = _bbox(mask)
    if bbox is None:
        axis = int(np.argmin(shape))
        return axis, shape[axis] // 2

    mins, maxs = bbox
    extents = maxs - mins + 1
    axis = int(np.argmin(extents))
    index = int(round((mins[axis] + maxs[axis]) / 2.0))
    return axis, index


def _window(vol: np.ndarray) -> tuple[float, float]:
    """Compute robust display intensity limits for image data.
    
    Args:
        vol (np.ndarray): Vol value used by the operation.
    
    Returns:
        tuple[float, float]: Tuple containing the values described by the return annotation.
        """
    values = vol[np.isfinite(vol)]
    values = values[np.abs(values) > 1e-6]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [1, 99])
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def _show_base(
    ax: plt.Axes, vol: np.ndarray, axis: int, index: int, title: str
) -> None:
    """Draw a grayscale reference slice on an axis.
    
    Args:
        ax (plt.Axes): Matplotlib axes object to draw on.
        vol (np.ndarray): Vol value used by the operation.
        axis (int): Axis index along which to operate.
        index (int): Voxel or candidate index used by the operation.
        title (str): Title to display on the generated plot.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    vmin, vmax = _window(vol)
    ax.imshow(_take_slice(vol, axis, index), cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _show_slab(
    ax: plt.Axes, vol: np.ndarray, axis: int, index: int, title: str, mode: str
) -> None:
    """Draw a slab slice using label or intensity display settings.
    
    Args:
        ax (plt.Axes): Matplotlib axes object to draw on.
        vol (np.ndarray): Vol value used by the operation.
        axis (int): Axis index along which to operate.
        index (int): Voxel or candidate index used by the operation.
        title (str): Title to display on the generated plot.
        mode (str): Processing mode controlling which images, labels, or metric are used.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    if mode == "label":
        ax.imshow(_take_slice(vol, axis, index), cmap="tab20", interpolation="nearest")
    else:
        vmin, vmax = _window(vol)
        ax.imshow(_take_slice(vol, axis, index), cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _contour(ax: plt.Axes, mask: np.ndarray, axis: int, index: int, color: str) -> None:
    """Draw a mask contour on an image axis.
    
    Args:
        ax (plt.Axes): Matplotlib axes object to draw on.
        mask (np.ndarray): Binary or label mask used to limit the operation.
        axis (int): Axis index along which to operate.
        index (int): Voxel or candidate index used by the operation.
        color (str): Color value used by the operation.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    sl = _take_slice(mask.astype(np.uint8), axis, index)
    if sl.any():
        ax.contour(sl, levels=[0.5], colors=[color], linewidths=1.2)


def _overlay(
    ax: plt.Axes, vol: np.ndarray, mask: np.ndarray, axis: int, index: int, cmap: str
) -> None:
    """Overlay slab data onto a reference slice for comparison.
    
    Args:
        ax (plt.Axes): Matplotlib axes object to draw on.
        vol (np.ndarray): Vol value used by the operation.
        mask (np.ndarray): Binary or label mask used to limit the operation.
        axis (int): Axis index along which to operate.
        index (int): Voxel or candidate index used by the operation.
        cmap (str): Matplotlib colormap used for display.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    sl = np.ma.masked_where(
        ~_take_slice(mask, axis, index),
        _take_slice(vol, axis, index),
    )
    if sl.count() > 0:
        ax.imshow(sl, cmap=cmap, alpha=0.45)


def _reference_path(preprocessed_dir: Path, patient: str, scan: str, mode: str) -> Path:
    """Resolve the reference image path for a manifest entry.
    
    Args:
        preprocessed_dir (Path): Directory containing preprocessed patient data.
        patient (str): Patient identifier associated with outputs.
        scan (str): Scan name or scan phase to process.
        mode (str): Processing mode controlling which images, labels, or metric are used.
    
    Returns:
        Path: Path to the generated or resolved filesystem resource.
        """
    pre_dir = preprocessed_dir / patient / "PreMortem"
    if mode == "label":
        if scan.startswith("segment_"):
            scan = scan.removeprefix("segment_")
        return pre_dir / f"{scan}.nii.gz"
    return pre_dir / f"{scan}.nii.gz"


def _manifest_output_dir(
    manifest_path: Path, registration_dir: Path, output_dir: Path
) -> Path:
    """Resolve the figure output directory for a manifest file.
    
    Args:
        manifest_path (Path): Filesystem path used by the operation.
        registration_dir (Path): Directory path used by the operation.
        output_dir (Path): Directory where generated outputs are written.
    
    Returns:
        Path: Path to the generated or resolved filesystem resource.
        """
    return output_dir / manifest_path.relative_to(registration_dir).parent


def _slice_label(axis: int, index: int) -> str:
    """Format a human-readable label for a slice axis and index.
    
    Args:
        axis (int): Axis index along which to operate.
        index (int): Voxel or candidate index used by the operation.
    
    Returns:
        str: String path, label, mode, or identifier produced by the operation.
        """
    return f"axis={axis}, index={index}"


def _create_overview(
    entries: list[dict],
    ref: np.ndarray,
    masks: list[np.ndarray],
    out_dir: Path,
) -> Path:
    """Create an overview figure for one registration manifest entry.
    
    Args:
        entries (list[dict]): Entries value used by the operation.
        ref (np.ndarray): Ref value used by the operation.
        masks (list[np.ndarray]): Mask data used to constrain or describe the operation.
        out_dir (Path): Directory path used by the operation.
    
    Returns:
        Path: Path to the generated or resolved filesystem resource.
        """
    axis = 2
    union = np.logical_or.reduce(masks)
    areas = union.sum(axis=(0, 1))
    index = int(np.argmax(areas)) if np.any(areas) else ref.shape[axis] // 2

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
    title = (
        f"{entries[0]['patient']} {entries[0]['slab']} "
        f"{entries[0]['mode']}/{entries[0]['scan']}\n"
        f"axial overview, {_slice_label(axis, index)}"
    )
    _show_base(ax, ref, axis, index, title)
    for entry, mask, color in zip(entries, masks, COLORS):
        _contour(ax, mask, axis, index, color)
        coords = np.argwhere(_take_slice(mask, axis, index))
        if coords.size:
            y, x = coords.mean(axis=0)
            ax.text(
                x,
                y,
                str(entry["rank"]),
                color=color,
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
            )

    out_path = out_dir / "overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _create_comparison(
    entry: dict,
    ref: np.ndarray,
    vol: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    mode: str,
) -> Path:
    """Create a comparison figure across top registration candidates.
    
    Args:
        entry (dict): Entry value used by the operation.
        ref (np.ndarray): Ref value used by the operation.
        vol (np.ndarray): Vol value used by the operation.
        mask (np.ndarray): Binary or label mask used to limit the operation.
        out_dir (Path): Directory path used by the operation.
        mode (str): Processing mode controlling which images, labels, or metric are used.
    
    Returns:
        Path: Path to the generated or resolved filesystem resource.
        """
    axis, index = _slice_axis_and_index(mask, ref.shape)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)

    _show_base(
        axes[0],
        ref,
        axis,
        index,
        f"full brain MRI | {_slice_label(axis, index)}",
    )
    _contour(axes[0], mask, axis, index, COLORS[(entry["rank"] - 1) % len(COLORS)])

    _show_slab(
        axes[1],
        vol,
        axis,
        index,
        f"warped slab rank {entry['rank']} | {entry['score_name']}={entry['score']:.4g}",
        mode,
    )

    out_path = out_dir / f"rank_{entry['rank']:02d}_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def create_manifest_figures(
    manifest_path: Path,
    registration_dir: Path,
    preprocessed_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Create all diagnostic figures for one manifest file.
    
    Args:
        manifest_path (Path): Filesystem path used by the operation.
        registration_dir (Path): Directory path used by the operation.
        preprocessed_dir (Path): Directory containing preprocessed patient data.
        output_dir (Path): Directory where generated outputs are written.
    
    Returns:
        list[Path]: List containing the generated or resolved values.
        """
    entries = json.loads(manifest_path.read_text())
    if not entries:
        return []
    entries = entries[:5]

    patient = entries[0]["patient"]
    scan = entries[0]["scan"]
    mode = entries[0]["mode"]
    ref_path = _reference_path(preprocessed_dir, patient, scan, mode)
    if not ref_path.exists():
        print(f"missing reference image for {manifest_path}: {ref_path}")
        return []

    ref = _load(ref_path)
    warped = [_load(Path(entry["warped_volume"])) for entry in entries]
    masks = [_mask(vol) for vol in warped]

    out_dir = _manifest_output_dir(manifest_path, registration_dir, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = [_create_overview(entries, ref, masks, out_dir)]
    for entry, vol, mask in zip(entries, warped, masks):
        written.append(_create_comparison(entry, ref, vol, mask, out_dir, mode))
    return written


def create_all_figures(
    registration_dir: Path,
    preprocessed_dir: Path,
    output_dir: Path,
    limit: int | None = None,
) -> list[Path]:
    """Create registration diagnostic figures for all available manifests.
    
    Args:
        registration_dir (Path): Directory path used by the operation.
        preprocessed_dir (Path): Directory containing preprocessed patient data.
        output_dir (Path): Directory where generated outputs are written.
        limit (int | None): Optional limit value. Defaults to `None`.
    
    Returns:
        list[Path]: List containing the generated or resolved values.
        """
    manifests = sorted(registration_dir.glob("**/manifest.json"))
    if limit is not None:
        manifests = manifests[:limit]

    written = []
    for manifest_path in manifests:
        print(f"visualizing {manifest_path}")
        outputs = create_manifest_figures(
            manifest_path=manifest_path,
            registration_dir=registration_dir,
            preprocessed_dir=preprocessed_dir,
            output_dir=output_dir,
        )
        written.extend(outputs)
    return written


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for this script.
    
    Returns:
        argparse.Namespace: Result produced by the operation in the form described by the return annotation.
        """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration-dir", type=Path, default=Path("outputs/registration")
    )
    parser.add_argument("--preprocessed-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/registration_visualizations")
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    paths = create_all_figures(
        registration_dir=args.registration_dir,
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(f"wrote {len(paths)} figure(s) to {args.output_dir}")
