"""Shared plotting helpers for image section figures."""
import matplotlib.pyplot as plt
import numpy as np


def plot_sections(
    sections: list[dict],
    title: str,
    title_fontsize: int | None = None,
    label_fontsize: int | None = None,
    bg_colour: str = "white",
    cmap_name: str = "gray",
) -> tuple[plt.Figure, list]:
    """Takes in a list of sections in the form of dict of:
    
    Args:
        sections (list[dict]): Sections value used by the operation.
        title (str): Title to display on the generated plot.
        title_fontsize (int | None): Optional title fontsize value. Defaults to `None`.
        label_fontsize (int | None): Label data used by the operation.
        bg_colour (str): Optional bg colour value. Defaults to `'white'`.
        cmap_name (str): Optional cmap name value. Defaults to `'gray'`.
    
    Returns:
        tuple[plt.Figure, list]: Tuple containing the values described by the return annotation.
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

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(bg_colour)

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
            np.ma.masked_where(section["data"] == 0, section["data"]),
            cmap=cmap,
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
    return fig, axes
