"""Plotting helpers for slab registration diagnostics."""
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


def plot_first_position(
    out_png: Path,
    pre_t1c_path: Path,
    pre_seg_path: Path,
    post_t2_path: Path,
    post_seg_path: Path,
    init_path: Path,
):
    """Plot the first candidate slab position against reference labels."""
    pre = nib.load(pre_t1c_path)
    pre_seg = nib.load(pre_seg_path)
    post = nib.load(init_path)
    post_seg = nib.load(post_seg_path)

    post_center_vox = np.array(
        [[(post.shape[0] - 1) / 2, (post.shape[1] - 1) / 2, (post.shape[2] - 1) / 2]]
    )
    y_world = nib.affines.apply_affine(post.affine, post_center_vox)[0][1]
    inv = np.linalg.inv(pre.affine)
    pre_y = int(round(nib.affines.apply_affine(inv, [[0, y_world, 0]])[0][1]))
    pre_y = int(np.clip(pre_y, 0, pre.shape[1] - 1))
    post_y = post.shape[1] // 2

    pre_slice = np.asanyarray(pre.dataobj)[:, pre_y, :].T
    pre_seg_slice = np.asanyarray(pre_seg.dataobj)[:, pre_y, :].T
    post_slice = np.asanyarray(post.dataobj)[:, post_y, :].T
    post_seg_slice = np.asanyarray(post_seg.dataobj)[:, post_y, :].T

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    axes[0, 0].imshow(pre_slice, cmap="gray", origin="lower")
    axes[0, 0].imshow(
        np.ma.masked_where(pre_seg_slice == 0, pre_seg_slice),
        alpha=0.35,
        origin="lower",
        vmin=1,
        vmax=2,
    )
    axes[0, 0].set_title("Pre t1c + GM/WM")
    axes[0, 1].imshow(post_slice, cmap="gray", origin="lower")
    axes[0, 1].imshow(
        np.ma.masked_where(post_seg_slice == 0, post_seg_slice),
        alpha=0.35,
        origin="lower",
        vmin=1,
        vmax=2,
    )
    axes[0, 1].set_title("Post t2 + tissue labels")
    axes[1, 0].imshow(pre_seg_slice, origin="lower", vmin=0, vmax=2)
    axes[1, 0].set_title("Pre segmentation")
    axes[1, 1].imshow(post_seg_slice, origin="lower", vmin=0, vmax=2)
    axes[1, 1].set_title("Post segmentation")
    for ax in axes.ravel():
        ax.axis("off")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_metrics(df: pd.DataFrame, out_dir: Path):
    """Plot registration metric summaries from a results table."""
    plot_specs = [
        ("none", "MI and Dice without registration", "metrics_no_registration.png"),
        (
            "intensity_affine",
            "MI and Dice after intensity affine registration",
            "metrics_intensity_affine.png",
        ),
        (
            "segmentation_affine",
            "MI and Dice after segmentation affine registration",
            "metrics_segmentation_affine.png",
        ),
    ]
    for patient, patient_df in df.groupby("patient"):
        for method, title, name in plot_specs:
            sub = patient_df[patient_df["method"] == method].copy()
            if sub.empty:
                continue
            fig, axes = plt.subplots(
                2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
            )
            for slab, grp in sub.groupby("slab"):
                label = str(slab)
                grp = grp.sort_values("ap_mm")
                axes[0].plot(
                    grp["ap_mm"], grp["mi"], marker=".", linewidth=1, label=label
                )
                axes[1].plot(
                    grp["ap_mm"], grp["dice"], marker=".", linewidth=1, label=label
                )
            axes[0].set_title(f"{patient}: {title}")
            axes[0].set_ylabel("MI")
            axes[1].set_ylabel("Dice")
            axes[1].set_xlabel("A-P coordinate (mm)")
            axes[0].grid(alpha=0.25)
            axes[1].grid(alpha=0.25)
            axes[0].legend(fontsize=8, ncol=2)
            stem = Path(name).stem
            fig.savefig(out_dir / f"{patient}_{stem}.png", dpi=180)
            plt.close(fig)

    for method, title, name in plot_specs:
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue
        fig, axes = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
        )
        for key, grp in sub.groupby(["patient", "slab"]):
            label = f"{key[0]} {key[1]}"
            grp = grp.sort_values("ap_mm")
            axes[0].plot(grp["ap_mm"], grp["mi"], marker=".", linewidth=1, label=label)
            axes[1].plot(
                grp["ap_mm"], grp["dice"], marker=".", linewidth=1, label=label
            )
        axes[0].set_title(title)
        axes[0].set_ylabel("MI")
        axes[1].set_ylabel("Dice")
        axes[1].set_xlabel("A-P coordinate (mm)")
        axes[0].grid(alpha=0.25)
        axes[1].grid(alpha=0.25)
        axes[0].legend(fontsize=7, ncol=2)
        fig.savefig(out_dir / name, dpi=180)
        plt.close(fig)
