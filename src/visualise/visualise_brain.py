"""
Takes in a path to volume and saves the three slices at given point to png
"""

from pathlib import Path

import ants
import matplotlib.colors as c
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.visualise.plotters import plot_sections


def _get_point(volume: np.ndarray) -> tuple:
    """Choose a representative foreground point from a volume.
    
    Args:
        volume (np.ndarray): 3D image array to slice or visualise.
    
    Returns:
        tuple: Tuple containing the values described by the return annotation.
        """
    coords = np.argwhere(volume != 0)
    crop_start, crop_stop = coords.min(axis=0), coords.max(axis=0) + 1
    point = tuple(((crop_start + crop_stop) // 2).tolist())
    return point


def get_sections(
    image: ants.ANTsImage,
    point: tuple[int, int, int] | None = None,
    plot_crosshair: bool = True,
) -> list[dict]:
    """Extract orthogonal image sections for plotting.
    
    Args:
        image (ants.ANTsImage): Image volume to process.
        point (tuple[int, int, int] | None): Optional point value. Defaults to `None`.
        plot_crosshair (bool): Optional plot crosshair value. Defaults to `True`.
    
    Returns:
        list[dict]: List containing the generated or resolved values.
        """
    volume = image.numpy()

    spacing = np.array(image.spacing)
    coords = np.argwhere(volume != 0)
    crop_start, crop_stop = coords.min(axis=0), coords.max(axis=0) + 1

    if point is None:
        point = _get_point(volume)

    x, y, z = point
    sx, sy, sz = spacing
    x0, y0, z0 = crop_start
    x1, y1, z1 = crop_stop

    sections = [
        {
            "title": "a) Coronal",
            "data": volume[x0:x1, y, z0:z1].T,
            "extent": (x0 * sx, x1 * sx, z0 * sz, z1 * sz),
            "crosshair": (x * sx, z * sz) if plot_crosshair else (None, None),
        },
        {
            "title": "b) Sagittal",
            "data": volume[x, y0:y1, z0:z1].T,
            "extent": (y0 * sy, y1 * sy, z0 * sz, z1 * sz),
            "crosshair": (y * sy, z * sz) if plot_crosshair else (None, None),
        },
        {
            "title": "c) Axial",
            "data": volume[x0:x1, y0:y1, z].T,
            "extent": (x0 * sx, x1 * sx, y0 * sy, y1 * sy),
            "crosshair": (x * sx, y * sy) if plot_crosshair else (None, None),
        },
    ]

    return sections


def overlay_tumour(
    fig: plt.Figure,
    axes: list,
    base_image: ants.ANTsImage,
    label_path: Path,
    point: tuple[int, int, int] | None = None,
):
    """Overlay tumour labels on anatomical section images.
    
    Args:
        fig (plt.Figure): Matplotlib figure object to populate.
        axes (list): Axes value used by the operation.
        base_image (ants.ANTsImage): Image data used by the operation.
        label_path (Path): Filesystem path used by the operation.
        point (tuple[int, int, int] | None): Optional point value. Defaults to `None`.
    
    Returns:
        Any: Result produced by the operation in the form described by the return annotation.
        """
    label_image = ants.image_read(str(label_path))
    label_on_base = ants.resample_image_to_target(
        label_image,
        base_image,
        interp_type="nearestNeighbor",
    )
    label_np = label_on_base.numpy()
    base_np = base_image.numpy()
    spacing = np.array(base_image.spacing)

    coords = np.argwhere(base_np != 0)
    if coords.size == 0:
        raise ValueError("Base volume is empty")
    crop_start, crop_stop = coords.min(axis=0), coords.max(axis=0) + 1

    colors = {
        1: {"color": "red", "label": "NCR/NET", "alpha": 0.4},
        2: {"color": "yellow", "label": "O", "alpha": 0.4},
        4: {"color": "green", "label": "ET", "alpha": 0.4},
    }
    if point is None:
        point = tuple(((crop_start + crop_stop) // 2).tolist())

    x, y, z = point
    sx, sy, sz = spacing
    x0, y0, z0 = crop_start
    x1, y1, z1 = crop_stop
    label_sections = [
        {
            "data": label_np[x0:x1, y, z0:z1].T,
            "extent": (x0 * sx, x1 * sx, z0 * sz, z1 * sz),
        },
        {
            "data": label_np[x, y0:y1, z0:z1].T,
            "extent": (y0 * sy, y1 * sy, z0 * sz, z1 * sz),
        },
        {
            "data": label_np[x0:x1, y0:y1, z].T,
            "extent": (x0 * sx, x1 * sx, y0 * sy, y1 * sy),
        },
    ]

    legend_patches = []
    for ax, section in zip(axes, label_sections):
        label_slice = section["data"]
        h, w = label_slice.shape
        overlay = np.zeros((h, w, 4), dtype=float)

        for value, properties in colors.items():
            class_mask = label_slice == value
            rgb = c.to_rgb(properties["color"])

            overlay[class_mask, :3] = rgb
            overlay[class_mask, 3] = properties["alpha"]

        ax.imshow(
            overlay,
            interpolation="none",
            origin="lower",
            extent=section["extent"],
        )

    for properties in colors.values():
        patch = mpatches.Patch(
            color=properties["color"],
            label=properties["label"],
            alpha=properties["alpha"],
        )
        legend_patches.append(patch)

    fig.legend(
        handles=legend_patches,
        loc="lower left",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )
    return fig, axes


def visualise_brain(
    volume_path: Path,
    mask_path: Path | None = None,
    point: tuple[int, int, int] | None = None,
    fig_title: str | None = None,
    output_path: Path | None = None,
    output_dir_stem: str = "images",
    plot_crosshair: bool = True,
) -> None:
    """Create a brain visualisation with optional labels and tumour overlays.
    
    Args:
        volume_path (Path): Filesystem path used by the operation.
        mask_path (Path | None): Filesystem path used by the operation.
        point (tuple[int, int, int] | None): Optional point value. Defaults to `None`.
        fig_title (str | None): Optional fig title value. Defaults to `None`.
        output_path (Path | None): Filesystem path used by the operation.
        output_dir_stem (str): Directory path used by the operation.
        plot_crosshair (bool): Optional plot crosshair value. Defaults to `True`.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    image = ants.image_read(str(volume_path))

    parts = volume_path.parts
    patient, time, scan = parts[-3], parts[-2], parts[-1][:-7]

    sections = get_sections(image, point, plot_crosshair=plot_crosshair)

    if fig_title is None:
        fig_title = f"{patient.upper()} - {time} - {scan}"
    fig, axes = plot_sections(sections, fig_title)

    if mask_path is not None:
        fig, axes = overlay_tumour(fig, axes, image, mask_path, point)

    if output_path is None:
        output_path = Path(output_dir_stem) / patient / time / f"{scan}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for f in Path("./prep/IM008/PostMortem").rglob("*.nii.gz"):
        visualise_brain(
            f,
            output_dir_stem="images/IM008",
            plot_crosshair=False,
        )
