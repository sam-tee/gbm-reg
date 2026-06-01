"""Slab visualisation helpers for orthogonal image sections."""
from pathlib import Path

import ants
import numpy as np

from src.visualise.plotters import plot_sections


def get_sections(image: ants.ANTsImage, type: str) -> list[dict]:
    """Extract orthogonal image sections for plotting.
    
    Args:
        image (ants.ANTsImage): Image volume to process.
        type (str): Section orientation or image type to extract.
    
    Returns:
        list[dict]: List containing the generated or resolved values.
        """
    volume = image.numpy()

    spacing = np.array(image.spacing)

    coords = np.argwhere(volume != 0)
    crop_start, crop_stop = coords.min(axis=0), coords.max(axis=0) + 1

    point = tuple(((crop_start + crop_stop) // 2).tolist())

    x, y, z = point
    sx, sy, sz = spacing
    x0, y0, z0 = crop_start
    x1, y1, z1 = crop_stop

    sections = [
        {
            "title": type,
            "data": volume[x0:x1, y, z0:z1].T,
            "extent": (x0 * sx, x1 * sx, z0 * sz, z1 * sz),
            "crosshair": (x * sx, z * sz),
        },
    ]
    return sections


def get_comparison_sections(base_dir: Path, images: list[str]):
    """Plots middle coronal section of two images side by side
    
    Args:
        base_dir (Path): Directory containing the images to visualise.
        images (list[str]): Image filenames or keys to visualise.
    
    Returns:
        Any: Result produced by the operation in the form described by the return annotation.
        """
    sections = []
    for name in images:
        image = ants.image_read(str(base_dir / f"{name}.nii.gz"))
        sections.extend(get_sections(image, name))
    return sections


def visualise_slab(
    volume_dir: Path,
    images: list[str] = ["t1", "t2"],
    fig_title: str | None = None,
    output_path: Path | None = None,
) -> None:
    """Create a slab section visualisation from saved image files.
    
    Args:
        volume_dir (Path): Directory path used by the operation.
        images (list[str]): Image filenames or keys to visualise.
        fig_title (str | None): Optional fig title value. Defaults to `None`.
        output_path (Path | None): Filesystem path used by the operation.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """

    parts = volume_dir.parts
    patient, time, series = parts[-3], parts[-2], parts[-1]

    sections = get_comparison_sections(volume_dir, images)

    if fig_title is None:
        fig_title = f"{patient.upper()} - {time} - {series}"
    fig = plot_sections(sections, fig_title, title_fontsize=18, label_fontsize=16)

    if output_path is None:
        output_path = Path("./images") / patient / time / series / "comp.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    for f in Path("./preprocessed/CUH197/PostMortem/").iterdir():
        visualise_slab(f)
