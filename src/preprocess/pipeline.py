"""Preprocessing pipelines for patient scans and post-mortem slabs."""
from pathlib import Path

import ants

from src.classes import Dirs, PostMortemSeries, PreprocessConfig
from src.preprocess.ants_wrappers import flip_image, n4_bias_correct, resample
from src.preprocess.mask import exvivo_mask, skull_strip
from src.preprocess.normalise import normalise


def preprocess_invivo(
    scans: dict[str, Path],
    config: PreprocessConfig,
    dirs: Dirs,
    patient_name: str,
    timepoint: str,
):
    """Preprocesses a single invivo scan set made up of t1, t1c, t2, flair scans
    
    Args:
        scans (dict[str, Path]): Mapping of scan names to input image paths.
        config (PreprocessConfig): Configuration object or dictionary controlling the operation.
        dirs (Dirs): Directory path used by the operation.
        patient_name (str): Patient identifier to process.
        timepoint (str): Timepoint value used by the operation.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    out_dir = dirs.preprocessed / patient_name / timepoint
    out_dir.mkdir(exist_ok=True, parents=True)
    scans_images: dict[str, ants.ANTsImage] = {}
    for type, path in scans.items():
        try:
            image = ants.image_read(str(path))
        except FileNotFoundError:
            print(f"Error: {type} scan not found at {path}")
            continue
        image = n4_bias_correct(image, config.bias_shrink_factor)
        scans_images[type] = resample(image)
    fixed = scans_images["t1c"]
    scans_processed = {"t1c": fixed}
    print("Bias corrected scans")
    # register to t1c
    for type in ["t1", "t2", "flair"]:
        moving = scans_images.get(type, None)
        if moving is None:
            continue
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")
        scans_processed[type] = reg["warpedmovout"]
    print("Registered scans to each other")
    # skull strip
    brain_mask = skull_strip(scans_processed["t1c"], dirs.temp, config)
    print("Stripped Skull")
    for type, scan in scans_processed.items():
        image_final = normalise(scan, brain_mask, config)
        out_path = out_dir / f"{type}.nii.gz"
        ants.image_write(image_final, str(out_path))
        print(f"Saved: {out_path}")
    mask_path = out_dir / "brain_mask.nii.gz"
    ants.image_write(brain_mask, str(mask_path))
    print(f"Saved brain mask: {mask_path}")


def preprocess_exvivo(
    data: dict[str, PostMortemSeries],
    config: PreprocessConfig,
    dirs: Dirs,
    patient_name: str,
    timepoint: str,
):
    """Preprocesses all sets of exvivo scans
    
    Args:
        data (dict[str, PostMortemSeries]): Array data to process or display.
        config (PreprocessConfig): Configuration object or dictionary controlling the operation.
        dirs (Dirs): Directory path used by the operation.
        patient_name (str): Patient identifier to process.
        timepoint (str): Timepoint value used by the operation.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    for series_name, series in data.items():
        out_dir = dirs.preprocessed / patient_name / timepoint / series_name
        out_dir.mkdir(exist_ok=True, parents=True)
        scans_images = {}
        for type in ["t1", "t2"]:
            path = getattr(series, type)
            try:
                image = ants.image_read(str(path))
            except FileNotFoundError:
                print(f"Error: {type} scan not found at {path}")
                continue
            flipped_img = flip_image(image, series.flips)
            scans_images[type] = n4_bias_correct(flipped_img, config.bias_shrink_factor)
        print("Bias corrected images")
        reg = ants.registration(
            fixed=scans_images["t1"],
            moving=scans_images["t2"],
            type_of_transform="Rigid",
        )
        print("Registered images")
        scans_reg = {"t1": scans_images["t1"], "t2": reg["warpedmovout"]}
        tissue_mask = exvivo_mask(scans_reg["t2"], config)

        for type, scan in scans_reg.items():
            image_final = normalise(scan, tissue_mask, config)
            out_path = out_dir / f"{type}.nii.gz"
            ants.image_write(image_final, str(out_path))
            print(f"Saved: {out_path}")
        mask_path = out_dir / "brain_mask.nii.gz"
        ants.image_write(tissue_mask, str(mask_path))
        print(f"Saved brain mask: {mask_path}")
