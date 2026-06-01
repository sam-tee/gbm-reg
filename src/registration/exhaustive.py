"""Exhaustive initialisation utilities for SimpleITK-based registration experiments."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import SimpleITK as sitk

TUMOUR_LABELS = "tumour_labels.nii.gz"


def rigid_transform(
    fixed_image: sitk.Image, moving_image: sitk.Image, centre: bool = True
) -> sitk.Euler3DTransform:
    """Create a 3D Euler rigid transform from rotations and translations.
    
    Args:
        fixed_image (sitk.Image): Image data used by the operation.
        moving_image (sitk.Image): Image data used by the operation.
        centre (bool): Voxel coordinate around which to extract views.
    
    Returns:
        sitk.Euler3DTransform: Result produced by the operation in the form described by the return annotation.
        """
    transform = sitk.Euler3DTransform()
    if centre:
        transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            transform,
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    return transform


def binary_mask(mask: sitk.Image, labels: tuple[int, ...] | None = None) -> sitk.Image:
    """Build or transform binary mask data used by the workflow.
    
    Args:
        mask (sitk.Image): Binary or label mask used to limit the operation.
        labels (tuple[int, ...] | None): Label image or label values used by the operation.
    
    Returns:
        sitk.Image: Image object containing the processed image, mask, or labels.
        """
    if labels is None:
        out = mask != 0
    else:
        out = mask == labels[0]
        for label in labels[1:]:
            out = out | (mask == label)
    return sitk.Cast(out, sitk.sitkUInt8)


def tumour_mask(tumour_labels: sitk.Image) -> sitk.Image:
    """Build or transform binary mask data used by the workflow.
    
    Args:
        tumour_labels (sitk.Image): Tumour label image used to localise the tumour region.
    
    Returns:
        sitk.Image: Image object containing the processed image, mask, or labels.
        """
    return binary_mask(tumour_labels)


def calculate_optimized_steps(mask: sitk.Image, step_mm: float = 2.0) -> list[int]:
    """Choose exhaustive-search step counts from the physical mask size.
    
    Args:
        mask (sitk.Image): Binary or label mask used to limit the operation.
        step_mm (float): Step size in millimetres.
    
    Returns:
        list[int]: List containing the generated or resolved values.
        """
    def _get_phys_radius(mask):
        """Estimate the physical radius of a mask bounding box.
        
        Args:
            mask: Binary or label mask used to limit the operation.
        
        Returns:
            Any: Result produced by the operation in the form described by the return annotation.
            """
        mask = binary_mask(mask)
        spacing = np.array(mask.GetSpacing())
        shape_filter = sitk.LabelShapeStatisticsImageFilter()
        shape_filter.Execute(mask)
        if not shape_filter.HasLabel(1):
            raise ValueError("Cannot calculate search steps from an empty mask")

        bbox_voxels = np.array(shape_filter.GetBoundingBox(1)[3:])
        return bbox_voxels * spacing / 2.0

    steps = np.ceil(_get_phys_radius(mask) / step_mm)
    return [max(1, int(step)) for step in steps]


def exhaustive_initial_transform(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    search_mask: sitk.Image | None = None,
    rotation_step_deg: float = 10.0,
    rotation_steps: int = 5,
    step_size_mm: float = 2.0,
) -> sitk.Euler3DTransform:
    """Uses exhaustive optimiser in SimpleITK to find best initial starting
    
    Args:
        fixed_image (sitk.Image): Image data used by the operation.
        moving_image (sitk.Image): Image data used by the operation.
        fixed_mask (sitk.Image): Mask data used to constrain or describe the operation.
        moving_mask (sitk.Image): Mask data used to constrain or describe the operation.
        search_mask (sitk.Image | None): Mask data used to constrain or describe the operation.
        rotation_step_deg (float): Optional rotation step deg value. Defaults to `10.0`.
        rotation_steps (int): Optional rotation steps value. Defaults to `5`.
        step_size_mm (float): Optional step size mm value. Defaults to `2.0`.
    
    Returns:
        sitk.Euler3DTransform: Result produced by the operation in the form described by the return annotation.
        """
    fixed_mask = binary_mask(fixed_mask)
    moving_mask = binary_mask(moving_mask)
    if search_mask is None:
        search_mask = fixed_mask
    search_mask = binary_mask(search_mask)

    reg = sitk.ImageRegistrationMethod()

    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingPercentage(0.5)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricFixedMask(fixed_mask)
    reg.SetMetricMovingMask(moving_mask)

    reg.SetInitialTransform(rigid_transform(fixed_image, moving_image, centre=True))
    reg.SetInterpolator(sitk.sitkLinear)

    x_steps, y_steps, z_steps = calculate_optimized_steps(search_mask, step_size_mm)
    reg.SetOptimizerAsExhaustive(
        [0, rotation_steps // 2, 0, int(x_steps), int(y_steps), int(z_steps)]
    )
    reg.SetOptimizerScales(
        [0, np.deg2rad(rotation_step_deg), 0, step_size_mm, step_size_mm, step_size_mm]
    )

    final_transform = reg.Execute(fixed_image, moving_image)

    print("----------")
    print(f"Optimizer stop condition: {reg.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {reg.GetOptimizerIteration()}")
    print(f" Metric value: {reg.GetMetricValue()}")
    print(f" Final transform: {final_transform}")

    return final_transform


def preprocessed_paths(
    preprocessed: Path,
    patient: str,
    moving_slab: str,
    fixed_timepoint: str = "PreMortem",
    fixed_modality: str = "t1c",
    moving_modality: str = "t1",
) -> tuple[Path, Path, Path, Path]:
    """Resolve expected preprocessed image paths for one patient and slab.
    
    Args:
        preprocessed (Path): Preprocessed value used by the operation.
        patient (str): Patient identifier associated with outputs.
        moving_slab (str): Moving slab value used by the operation.
        fixed_timepoint (str): Optional fixed timepoint value. Defaults to `'PreMortem'`.
        fixed_modality (str): Optional fixed modality value. Defaults to `'t1c'`.
        moving_modality (str): Optional moving modality value. Defaults to `'t1'`.
    
    Returns:
        tuple[Path, Path, Path, Path]: Tuple containing the values described by the return annotation.
        """
    fixed_dir = preprocessed / patient / fixed_timepoint
    moving_dir = preprocessed / patient / "PostMortem" / moving_slab
    return (
        fixed_dir / f"{fixed_modality}.nii.gz",
        moving_dir / f"{moving_modality}.nii.gz",
        fixed_dir / TUMOUR_LABELS,
        moving_dir / "brain_mask.nii.gz",
    )


def run_preprocessed(
    preprocessed: Path,
    patient: str,
    moving_slab: str,
    fixed_timepoint: str = "PreMortem",
    fixed_modality: str = "t1c",
    moving_modality: str = "t1",
    output: Path | None = None,
    transform_output: Path | None = None,
    rotation_step_deg: float = 10.0,
    rotation_steps: int = 5,
    step_size_mm: float = 2.0,
) -> sitk.Image:
    """Run exhaustive registration using already-preprocessed inputs.
    
    Args:
        preprocessed (Path): Preprocessed value used by the operation.
        patient (str): Patient identifier associated with outputs.
        moving_slab (str): Moving slab value used by the operation.
        fixed_timepoint (str): Optional fixed timepoint value. Defaults to `'PreMortem'`.
        fixed_modality (str): Optional fixed modality value. Defaults to `'t1c'`.
        moving_modality (str): Optional moving modality value. Defaults to `'t1'`.
        output (Path | None): Optional output value. Defaults to `None`.
        transform_output (Path | None): Optional transform output value. Defaults to `None`.
        rotation_step_deg (float): Optional rotation step deg value. Defaults to `10.0`.
        rotation_steps (int): Optional rotation steps value. Defaults to `5`.
        step_size_mm (float): Optional step size mm value. Defaults to `2.0`.
    
    Returns:
        sitk.Image: Image object containing the processed image, mask, or labels.
        """
    fixed_path, moving_path, tumour_path, moving_mask_path = preprocessed_paths(
        preprocessed,
        patient,
        moving_slab,
        fixed_timepoint,
        fixed_modality,
        moving_modality,
    )
    for path in (fixed_path, moving_path, tumour_path, moving_mask_path):
        if not path.exists():
            raise FileNotFoundError(path)

    fixed_image = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving_image = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    fixed_tumour_mask = tumour_mask(sitk.ReadImage(str(tumour_path)))
    moving_mask = sitk.ReadImage(str(moving_mask_path))

    final_transform = exhaustive_initial_transform(
        fixed_image=fixed_image,
        moving_image=moving_image,
        fixed_mask=fixed_tumour_mask,
        moving_mask=moving_mask,
        search_mask=fixed_tumour_mask,
        rotation_step_deg=rotation_step_deg,
        rotation_steps=rotation_steps,
        step_size_mm=step_size_mm,
    )

    resampled_image = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )

    if output is None:
        output = (
            preprocessed
            / patient
            / "PostMortem"
            / moving_slab
            / f"{moving_modality}_resampled_to_{fixed_timepoint}_{fixed_modality}.nii.gz"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled_image, str(output))

    if transform_output is not None:
        transform_output.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(final_transform, str(transform_output))

    print(f" Saved resampled volume: {output}")
    if transform_output is not None:
        print(f" Saved transform: {transform_output}")

    return resampled_image


def parse_args() -> ArgumentParser:
    """Build and return the command-line argument parser.
    
    Returns:
        ArgumentParser: Result produced by the operation in the form described by the return annotation.
        """
    parser = ArgumentParser(
        description="Run tumour-ROI exhaustive registration on preprocessed data."
    )
    parser.add_argument("patient")
    parser.add_argument("moving_slab")
    parser.add_argument("--preprocessed", type=Path, default=Path("preprocessed"))
    parser.add_argument("--fixed-timepoint", default="PreMortem")
    parser.add_argument("--fixed-modality", default="t1c")
    parser.add_argument("--moving-modality", default="t1")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the resampled moving volume. Defaults inside the slab folder.",
    )
    parser.add_argument(
        "--transform-output",
        type=Path,
        help="Optional path for the final transform.",
    )
    parser.add_argument("--rotation-step-deg", type=float, default=10.0)
    parser.add_argument("--rotation-steps", type=int, default=5)
    parser.add_argument("--step-size-mm", type=float, default=2.0)
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    run_preprocessed(**vars(args))
