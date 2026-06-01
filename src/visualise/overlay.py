"""Overlay visualisation helpers for comparing registered image volumes."""
from pathlib import Path

import ants
import matplotlib.pyplot as plt
import numpy as np


def _slice_outline(mask: np.ndarray) -> np.ndarray:
    """Return the border pixels of a 2D binary mask.
    
    Args:
        mask (np.ndarray): Binary or label mask used to limit the operation.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~eroded


def _get_overlay_sections(
    base_image: ants.ANTsImage,
    label_image: ants.ANTsImage,
    outline: bool,
    point: tuple[int, int, int] | None,
) -> list[dict]:
    """Collect aligned sections needed for overlay visualisation.
    
    Args:
        base_image (ants.ANTsImage): Image data used by the operation.
        label_image (ants.ANTsImage): Image data used by the operation.
        outline (bool): Outline value used by the operation.
        point (tuple[int, int, int] | None): Point value used by the operation.
    
    Returns:
        list[dict]: List containing the generated or resolved values.
        """
    base_volume = base_image.numpy()
    label_on_base = ants.resample_image_to_target(
        label_image,
        base_image,
        interp_type="nearestNeighbor",
    )
    label_volume = label_on_base.numpy()

    spacing = np.array(base_image.spacing)

    coords = np.argwhere((base_volume != 0) | (label_volume != 0))
    if coords.size == 0:
        raise ValueError("Base and label volumes are both empty")

    crop_start, crop_stop = coords.min(axis=0), coords.max(axis=0) + 1
    if point is None:
        point = tuple(((crop_start + crop_stop) // 2).tolist())

    x, y, z = point
    sx, sy, sz = spacing
    x0, y0, z0 = crop_start
    x1, y1, z1 = crop_stop

    sections = [
        {
            "title": "a) Coronal",
            "base": base_volume[x0:x1, y, z0:z1].T,
            "label": label_volume[x0:x1, y, z0:z1].T,
            "extent": (x0 * sx, x1 * sx, z0 * sz, z1 * sz),
            "crosshair": (x * sx, z * sz),
            "slice_position": y * sy,
        },
        {
            "title": "b) Sagittal",
            "base": base_volume[x, y0:y1, z0:z1].T,
            "label": label_volume[x, y0:y1, z0:z1].T,
            "extent": (y0 * sy, y1 * sy, z0 * sz, z1 * sz),
            "crosshair": (y * sy, z * sz),
            "slice_position": x * sx,
        },
        {
            "title": "c) Axial",
            "base": base_volume[x0:x1, y0:y1, z].T,
            "label": label_volume[x0:x1, y0:y1, z].T,
            "extent": (x0 * sx, x1 * sx, y0 * sy, y1 * sy),
            "crosshair": (x * sx, y * sy),
            "slice_position": z * sz,
        },
    ]

    if outline:
        for section in sections:
            section["label"] = _slice_outline(section["label"] != 0)

    return sections


def _plot_overlay_sections(
    sections: list[dict],
    title: str,
    opacity: float,
    spatial_note: str | None = None,
    title_fontsize: int | None = None,
    label_fontsize: int | None = None,
) -> plt.Figure:
    """Render overlay section panels to a matplotlib figure.
    
    Args:
        sections (list[dict]): Sections value used by the operation.
        title (str): Title to display on the generated plot.
        opacity (float): Opacity value used by the operation.
        spatial_note (str | None): Optional spatial note value. Defaults to `None`.
        title_fontsize (int | None): Optional title fontsize value. Defaults to `None`.
        label_fontsize (int | None): Label data used by the operation.
    
    Returns:
        plt.Figure: Result produced by the operation in the form described by the return annotation.
        """
    max_plot_width_in = 12
    gutter_in = 0.45
    left_margin_in = 0.35
    right_margin_in = 0.25
    bottom_margin_in = 0.65
    label_y_in = 0.18
    top_margin_in = 0.6

    plot_widths = [
        abs(section["extent"][1] - section["extent"][0]) for section in sections
    ]
    plot_heights = [
        abs(section["extent"][3] - section["extent"][2]) for section in sections
    ]
    available_width_in = (
        max_plot_width_in
        - left_margin_in
        - right_margin_in
        - gutter_in * (len(sections) - 1)
    )
    scale = available_width_in / sum(plot_widths)
    plot_widths_in = [width * scale for width in plot_widths]
    plot_heights_in = [height * scale for height in plot_heights]
    fig_width_in = (
        left_margin_in
        + sum(plot_widths_in)
        + gutter_in * (len(sections) - 1)
        + right_margin_in
    )
    fig_height_in = bottom_margin_in + max(plot_heights_in) + top_margin_in

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    fig.suptitle(title, fontsize=title_fontsize)
    if spatial_note is not None:
        fig.text(
            0.5,
            0.94,
            spatial_note,
            ha="center",
            va="top",
            fontsize=8,
            color="0.35",
        )

    base_cmap = plt.get_cmap("gray").copy()
    base_cmap.set_bad("white")
    label_cmap = plt.get_cmap("autumn").copy()
    label_cmap.set_bad(alpha=0)

    axes = []
    left_in = left_margin_in
    for width_in, height_in in zip(plot_widths_in, plot_heights_in):
        axes.append(
            fig.add_axes(
                [
                    left_in / fig_width_in,
                    bottom_margin_in / fig_height_in,
                    width_in / fig_width_in,
                    height_in / fig_height_in,
                ]
            )
        )
        left_in += width_in + gutter_in

    for axis, section in zip(axes, sections):
        axis.imshow(
            np.ma.masked_where(section["base"] == 0, section["base"]),
            cmap=base_cmap,
            origin="lower",
            extent=section["extent"],
        )
        axis.imshow(
            np.ma.masked_where(section["label"] == 0, section["label"]),
            cmap=label_cmap,
            alpha=opacity,
            origin="lower",
            extent=section["extent"],
        )
        cx, cy = section.get("crosshair", (None, None))
        if (cx is not None) and (cy is not None):
            axis.axvline(cx, color="tab:red", linewidth=1, alpha=0.6)
            axis.axhline(cy, color="tab:red", linewidth=1, alpha=0.6)
        axis.set_axis_off()
        axis.set_aspect("equal")
        axis_center_x = axis.get_position().x0 + axis.get_position().width / 2
        fig.text(
            axis_center_x,
            label_y_in / fig_height_in,
            section["title"],
            ha="center",
            va="top",
            fontsize=label_fontsize,
        )

    return fig


def visualise_overlay(
    base_path: Path,
    label_path: Path,
    opacity: float,
    outline: bool = False,
    point: tuple[int, int, int] | None = None,
) -> plt.Figure:
    """Takes in paths to NifTi volumes for comparison volume and label volume and visualises both with given opacity
    
    Args:
        base_path (Path): Filesystem path used by the operation.
        label_path (Path): Filesystem path used by the operation.
        opacity (float): Opacity value used by the operation.
        outline (bool): Optional outline value. Defaults to `False`.
        point (tuple[int, int, int] | None): Optional point value. Defaults to `None`.
    
    Returns:
        plt.Figure: Result produced by the operation in the form described by the return annotation.
        """
    if not 0 <= opacity <= 1:
        raise ValueError(f"opacity must be between 0 and 1, got {opacity}")

    base_image = ants.image_read(str(base_path))
    label_image = ants.image_read(str(label_path))

    sections = _get_overlay_sections(base_image, label_image, outline, point)
    spatial_note = (
        "Label resampled to base grid | "
        f"origin={tuple(round(v, 2) for v in base_image.origin)} | "
        f"spacing={tuple(round(v, 2) for v in base_image.spacing)}"
    )

    parts = base_path.parts
    if len(parts) >= 3:
        patient, time, scan = parts[-3], parts[-2], parts[-1][:-7]
        fig_title = f"{patient.upper()} - {time} - {scan} overlay"
    else:
        fig_title = f"{base_path.name} + {label_path.name}"

    return _plot_overlay_sections(sections, fig_title, opacity, spatial_note)
