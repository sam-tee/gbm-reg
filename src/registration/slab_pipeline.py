"""Orchestration for coronal post-mortem slab registration."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import nibabel as nib
import numpy as np
import pandas as pd

from src.registration.slab_io import copy_transforms, load_ants
from src.registration.slab_masks import (
    hemisphere_center_x,
    nib_world_bounds,
    oedema_ap_range_brain_mask,
    patient_ap_positions,
    save_roi_masks,
    set_image_center_world,
    tumour_label_mask,
)
from src.registration.slab_metrics import metrics
from src.registration.slab_registration import (
    generalized_dice_affine_registration,
    safe_registration,
)
from src.registration.slab_visualise import plot_first_position, plot_metrics


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem paths used by the slab registration pipeline.
    
    Attributes:
        data (Path): Configuration or state value used by the pipeline.
        output (Path): Configuration or state value used by the pipeline.
        """
    data: Path
    output: Path


def _postmortem_tissue_label_path(slab_dir: Path) -> Path | None:
    """Return the post-mortem tissue labels created by segmentation."""
    path = slab_dir / "segment.nii.gz"
    return path if path.exists() else None


def run(
    paths: Paths,
    patients: list[str] | None = None,
    max_positions: int | None = None,
    skip_registration: bool = False,
    ap_step_mm: float = 5.0,
    masks_only: bool = False,
):
    """Run slab registration for selected patients.

    Args:
        paths (Paths): Resolved input and output directories for the pipeline.
        patients (list[str] | None): Optional patient identifiers to process.
        max_positions (int | None): Optional maximum number of candidate AP positions.
        skip_registration (bool): Whether to skip optimisation and only prepare masks/slabs.
        ap_step_mm (float): Candidate AP position spacing in millimetres.
        masks_only (bool): Whether to stop after generating ROI masks.

    Returns:
        None: This function returns `None`; masks, transforms, metrics, and figures are written to disk.
    """
    rows = []
    all_patient_dirs = sorted([p for p in paths.data.iterdir() if p.is_dir()])
    if patients:
        wanted = set(patients)
        all_patient_dirs = [p for p in all_patient_dirs if p.name in wanted]

    for patient_dir in all_patient_dirs:
        pid = patient_dir.name
        pre_dir = patient_dir / "PreMortem"
        patient_out = paths.output / pid
        transforms_dir = patient_out / "transforms"
        init_dir = patient_out / "initialized_slabs"
        patient_out.mkdir(parents=True, exist_ok=True)
        init_dir.mkdir(parents=True, exist_ok=True)

        pre_t1c = pre_dir / "t1c.nii.gz"
        pre_brain = pre_dir / "brain_mask.nii.gz"
        pre_tumour = pre_dir / "tumour_labels.nii.gz"
        pre_tissue = pre_dir / "tissue_labels.nii.gz"
        pre_oedema_mask = tumour_label_mask(
            pre_tumour, patient_out / "premortem_oedema_label2_mask.nii.gz", label=2
        )
        ap_band_mask, ap_min, ap_max = oedema_ap_range_brain_mask(
            pre_tumour,
            pre_brain,
            patient_out / "premortem_full_brain_oedema_ap_range_mask.nii.gz",
            label=2,
        )
        positions = patient_ap_positions(pre_tumour, ap_step_mm)
        if max_positions:
            positions = positions[:max_positions]
        x_center = hemisphere_center_x(pre_brain, pre_tumour)
        pre_brain_img = nib.load(pre_brain)
        brain_min, brain_max = nib_world_bounds(
            pre_brain_img, np.asanyarray(pre_brain_img.dataobj) > 0
        )
        z_center = float(0.5 * (brain_min[2] + brain_max[2]))

        pre_t1c_ants = load_ants(pre_t1c)
        pre_seg_ants = load_ants(pre_tissue, pixeltype="unsigned char")

        metadata = {
            "patient": pid,
            "ap_step_mm": ap_step_mm,
            "oedema_ap_min_mm": ap_min,
            "oedema_ap_max_mm": ap_max,
            "full_brain_oedema_ap_range_mask": str(ap_band_mask),
            "ap_positions_mm": positions.tolist(),
            "hemisphere_center_x_mm": x_center,
        }
        (patient_out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
        if masks_only:
            continue

        for slab_t2 in sorted((patient_dir / "PostMortem").glob("*/t2.nii.gz")):
            slab = slab_t2.parent.name
            post_seg_path = _postmortem_tissue_label_path(slab_t2.parent)
            if post_seg_path is None:
                print(
                    f"[{pid}/{slab}] missing post-mortem tissue labels, skipping. "
                    "Run segmentation before registration."
                )
                continue
            original_post = nib.load(slab_t2)

            first_plot_done = False
            for ap_mm in positions:
                tag = f"{pid}_{slab}_ap{ap_mm:+07.2f}".replace(".", "p")
                init_t2_path = init_dir / f"{tag}_t2.nii.gz"
                init_seg_path = init_dir / f"{tag}_gmwm.nii.gz"
                roi_post_path = init_dir / f"{tag}_roi_postspace.nii.gz"
                roi_pre_path = init_dir / f"{tag}_roi_premortem_space.nii.gz"

                center = np.array([x_center, ap_mm, z_center], dtype=float)
                nib.save(set_image_center_world(original_post, center), init_t2_path)
                nib.save(
                    set_image_center_world(nib.load(post_seg_path), center),
                    init_seg_path,
                )

                if not first_plot_done:
                    plot_first_position(
                        patient_out / f"{slab}_first_position_check.png",
                        pre_t1c,
                        pre_tissue,
                        slab_t2,
                        post_seg_path,
                        init_t2_path,
                    )
                    first_plot_done = True

                post_t2 = load_ants(init_t2_path)
                post_seg = load_ants(init_seg_path, pixeltype="unsigned char")
                _, _, roi_voxels = save_roi_masks(
                    init_t2_path, pre_t1c, pre_oedema_mask, roi_post_path, roi_pre_path
                )
                roi_mask = load_ants(roi_post_path, pixeltype="unsigned char")
                mi, dice = metrics(
                    post_t2, post_seg, pre_t1c_ants, pre_seg_ants, roi_mask
                )
                rows.append(
                    {
                        "patient": pid,
                        "slab": slab,
                        "ap_mm": ap_mm,
                        "method": "none",
                        "mi": mi,
                        "dice": dice,
                        "roi_postspace": str(roi_post_path),
                        "roi_premortem_space": str(roi_pre_path),
                        "roi_voxels": roi_voxels,
                        "transform_files": "",
                    }
                )

                if skip_registration:
                    continue

                print(f"[{pid}/{slab}] AP {ap_mm:.2f}: affine intensity")
                intensity_prefix = transforms_dir / f"{tag}_intensity_"
                reg_i = safe_registration(
                    post_t2, pre_t1c_ants, "mattes", intensity_prefix, is_seg=False
                )
                copied_i = copy_transforms(
                    reg_i["fwdtransforms"], transforms_dir, f"{tag}_intensity"
                )
                mi_i, dice_i = metrics(
                    post_t2,
                    post_seg,
                    pre_t1c_ants,
                    pre_seg_ants,
                    roi_mask,
                    reg_i["fwdtransforms"],
                )
                rows.append(
                    {
                        "patient": pid,
                        "slab": slab,
                        "ap_mm": ap_mm,
                        "method": "intensity_affine",
                        "mi": mi_i,
                        "dice": dice_i,
                        "roi_postspace": str(roi_post_path),
                        "roi_premortem_space": str(roi_pre_path),
                        "roi_voxels": roi_voxels,
                        "transform_files": "|".join(copied_i),
                    }
                )

                print(f"[{pid}/{slab}] AP {ap_mm:.2f}: affine segmentation")
                seg_prefix = transforms_dir / f"{tag}_segmentation_"
                reg_s = generalized_dice_affine_registration(
                    post_seg, pre_seg_ants, roi_mask, seg_prefix
                )
                copied_s = copy_transforms(
                    reg_s["fwdtransforms"], transforms_dir, f"{tag}_segmentation"
                )
                mi_s, dice_s = metrics(
                    post_t2,
                    post_seg,
                    pre_t1c_ants,
                    pre_seg_ants,
                    roi_mask,
                    reg_s["fwdtransforms"],
                )
                rows.append(
                    {
                        "patient": pid,
                        "slab": slab,
                        "ap_mm": ap_mm,
                        "method": "segmentation_affine",
                        "mi": mi_s,
                        "dice": dice_s,
                        "roi_postspace": str(roi_post_path),
                        "roi_premortem_space": str(roi_pre_path),
                        "roi_voxels": roi_voxels,
                        "transform_files": "|".join(copied_s),
                    }
                )

                pd.DataFrame(rows).to_csv(
                    paths.output / "slab_metrics.csv", index=False
                )

        (patient_out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    df = pd.DataFrame(rows)
    if masks_only:
        return df
    df.to_csv(paths.output / "slab_metrics.csv", index=False)
    plot_metrics(df, paths.output)
    return df


def main():
    """Run the script entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--ap-step-mm", type=float, default=5.0)
    parser.add_argument("--masks-only", action="store_true")
    parser.add_argument("--skip-registration", action="store_true")
    args = parser.parse_args()
    run(
        Paths(args.data, args.output),
        args.patients,
        args.max_positions,
        args.skip_registration,
        args.ap_step_mm,
        args.masks_only,
    )


if __name__ == "__main__":
    main()
