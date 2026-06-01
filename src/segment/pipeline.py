"""Segmentation pipelines for in-vivo and ex-vivo image sets."""

import shutil
from pathlib import Path

from src.classes import Dirs, SegmentConfig
from src.segment.gmm import run_gmm
from src.segment.medsam2 import run_medsam2_mask, run_medsam2_tumour_labels
from src.segment.swin_unetr import run_swin_unetr
from src.segment.synthseg import run_synthseg


def segment_invivo(
    scans: dict[str, Path],
    config: SegmentConfig,
    dirs: Dirs,
    patient_name: str,
    timepoint: str,
) -> None:
    """Segments a single invivo timepoint for a single patient

    Args:
        scans (dict[str, Path]): Mapping of scan names to input image paths.
        config (SegmentConfig): Configuration object or dictionary controlling the operation.
        dirs (Dirs): Directory path used by the operation.
        patient_name (str): Patient identifier to process.
        timepoint (str): Timepoint value used by the operation.

    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
    """
    out_dir = dirs.preprocessed / patient_name / timepoint
    out_dir.mkdir(exist_ok=True, parents=True)
    synthseg_scan = scans.get(config.synthseg_scan)
    if synthseg_scan is None:
        raise ValueError("Error: synthseg_scan must be set in config.toml")
    run_synthseg(synthseg_scan, out_dir, config.synthseg_flags)
    if config.swin_unetr_model_path is None:
        raise ValueError("Error: swin_unetr_model_path must be set in config.toml")
    run_swin_unetr(scans, config.swin_unetr_model_path, out_dir, config.device)


def segment_exvivo(
    patient_name: str,
    slab_name: str,
    config: SegmentConfig,
    dirs: Dirs,
) -> None:
    """Segments a single slab's scans

    Args:
        patient_name (str): Patient identifier to process.
        slab_name (str): Slab name value used by the operation.
        config (SegmentConfig): Configuration object or dictionary controlling the operation.
        dirs (Dirs): Directory path used by the operation.

    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
    """
    base_dir = dirs.preprocessed / patient_name / "PostMortem" / slab_name
    mask_path = base_dir / "brain_mask.nii.gz"
    algorithm = config.exvivo_label_algorithm.lower()

    if algorithm in {"medsam", "medsam2"}:
        prompt_scan = config.medsam2_prompt_scan.lower()
        if prompt_scan not in {"t1", "t2"}:
            raise ValueError(
                "segmentation.medsam2_prompt_scan must be one of ['t1', 't2']"
            )
        mask_path = run_medsam2_mask(
            image_path=base_dir / f"{prompt_scan}.nii.gz",
            prompt_mask_path=base_dir / "brain_mask.nii.gz",
            output_path=base_dir / "medsam2_mask.nii.gz",
            checkpoint=config.medsam2_checkpoint,
            model_cfg=config.medsam2_model_cfg,
            repo_path=config.medsam2_repo_path,
            device=config.device,
            slice_axis=config.medsam2_slice_axis,
            bbox_margin=config.medsam2_bbox_margin,
            image_size=config.medsam2_image_size,
        )
        if config.medsam2_tumour_labels:
            run_medsam2_tumour_labels(
                image_path=base_dir / f"{prompt_scan}.nii.gz",
                tissue_mask_path=mask_path,
                output_path=base_dir / "tumour_labels.nii.gz",
                checkpoint=config.medsam2_checkpoint,
                model_cfg=config.medsam2_model_cfg,
                repo_path=config.medsam2_repo_path,
                device=config.device,
                slice_axis=config.medsam2_slice_axis,
                bbox_margin=config.medsam2_bbox_margin,
                image_size=config.medsam2_image_size,
                tumour_label=config.medsam2_tumour_label,
                tumour_prompt_percentile=config.medsam2_tumour_prompt_percentile,
                tumour_prompt_min_fraction=config.medsam2_tumour_prompt_min_fraction,
            )
    elif algorithm != "gmm":
        raise ValueError(
            "segmentation.exvivo_label_algorithm must be one of ['gmm', 'medsam2']"
        )

    run_gmm(
        base_dir / "t1.nii.gz",
        mask_path,
        base_dir / "segment_t1.nii.gz",
        config.gmm_classes,
    )
    run_gmm(
        base_dir / "t2.nii.gz",
        mask_path,
        base_dir / "segment.nii.gz",
        config.gmm_classes,
    )
    shutil.copy2(base_dir / "segment.nii.gz", base_dir / "segment_t2.nii.gz")
