"""
Inputs:
    PreMortem:
        FreeSurfer labels and BraTS-style tumour label
    PostMortem:
         3 channel labels: (gm, wm, tumour)

Registers the PreMortem full brain volumes and the post mortem partial volumes based
on the segmentation labels.
The post-mortem volumes are 10mm thick volumes of the hemisphere with tumour, taken
in the coronal plane.

Initialise the post-mortem slab at different points in the tumour ROI and perform a
registration to fit.
Use an overlap-based metric and save the top 5 volumes and their overlap scores.
"""

import argparse
import json
import tempfile
from pathlib import Path

import ants
import numpy as np

from src.classes import Dirs, RegistrationConfig
from src.misc import load_toml

RAS = "RAS"
TOP_N = 5
NUM_INITS = 5
SLICE_THICKNESS_MM = 10.0
SLICE_STEP_MM = 2.0

# FreeSurfer label IDs for left and right cerebrum
LEFT_CEREBRUM_LABELS = [2, 3, 10, 11, 12, 13, 17, 18, 26, 28]
RIGHT_CEREBRUM_LABELS = [41, 42, 49, 50, 51, 52, 53, 54, 58, 60]

# GMM parameters for post-mortem tissue segmentation
POSTMORTEM_GMM_CLASSES = 3


# ─── geometry helpers ─────────────────────────────────────────────────────────


def _reorient(img: ants.ANTsImage) -> ants.ANTsImage:
    """Reorient an ANTs image to RAS.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    return ants.reorient_image2(img, RAS)


def _get_bbox(labels: ants.ANTsImage) -> tuple:
    """Return (x_min, y_min, z_min, x_max, y_max, z_max) of non-zero voxels.
    
    Args:
        labels (ants.ANTsImage): Label image or label values used by the operation.
    
    Returns:
        tuple: Tuple containing the values described by the return annotation.
        """
    coords = np.argwhere(labels.numpy() > 0)
    if len(coords) == 0:
        return (0, 0, 0, 0, 0, 0)
    return *coords.min(axis=0), *coords.max(axis=0)


def _get_hemisphere(tumour_labels: ants.ANTsImage, seg_labels: ants.ANTsImage) -> str:
    """Determine tumour hemisphere from tumour bbox vs brain bbox along the X axis.
    
    Args:
        tumour_labels (ants.ANTsImage): Tumour label image used to localise the tumour region.
        seg_labels (ants.ANTsImage): Segmentation label image used to identify anatomy.
    
    Returns:
        str: String path, label, mode, or identifier produced by the operation.
        """
    tumour_bbox = _get_bbox(tumour_labels)
    tumour_x_mid = (tumour_bbox[0] + tumour_bbox[3]) / 2.0
    brain_bbox = _get_bbox(seg_labels)
    brain_x_mid = (brain_bbox[0] + brain_bbox[3]) / 2.0
    return "left" if tumour_x_mid > brain_x_mid else "right"


def _get_cerebrum_mask(seg_labels: ants.ANTsImage, hemisphere: str) -> ants.ANTsImage:
    """Binary mask of the left or right cerebrum from a FreeSurfer label image.
    
    Args:
        seg_labels (ants.ANTsImage): Segmentation label image used to identify anatomy.
        hemisphere (str): Hemisphere value used by the operation.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    labels = LEFT_CEREBRUM_LABELS if hemisphere == "left" else RIGHT_CEREBRUM_LABELS
    mask = np.isin(seg_labels.numpy(), labels).astype(np.float32)
    return ants.new_image_like(seg_labels, mask)


def _roi_to_tumour_side(
    full_brain: ants.ANTsImage,
    tumour_labels: ants.ANTsImage,
    seg_labels: ants.ANTsImage,
) -> ants.ANTsImage:
    """Mask full-brain image to the tumour-side cerebrum only.
    
    Args:
        full_brain (ants.ANTsImage): Full brain value used by the operation.
        tumour_labels (ants.ANTsImage): Tumour label image used to localise the tumour region.
        seg_labels (ants.ANTsImage): Segmentation label image used to identify anatomy.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    hemi = _get_hemisphere(tumour_labels, seg_labels)
    cerebrum_mask = _get_cerebrum_mask(seg_labels, hemi)
    return ants.mask_image(full_brain, cerebrum_mask)


def _make_virtual_slices(
    roi: ants.ANTsImage,
    tumour_bbox: tuple,
    num_inits: int,
    slice_thickness_mm: float,
    slice_step_mm: float,
) -> list:
    """Crop virtual A-P (Y-axis) slices out of *roi*, stepping along the Y axis
    
    Args:
        roi (ants.ANTsImage): Roi value used by the operation.
        tumour_bbox (tuple): Tumour bbox value used by the operation.
        num_inits (int): Num inits value used by the operation.
        slice_thickness_mm (float): Slice data or slice identifier used by the operation.
        slice_step_mm (float): Slice data or slice identifier used by the operation.
    
    Returns:
        list: List containing the generated or resolved values.
        """
    _, y_min, _, _, y_max, _ = tumour_bbox
    y_spacing = roi.spacing[1]
    thickness_vox = round(slice_thickness_mm / y_spacing)
    step_vox = max(round(slice_step_mm / y_spacing), 1)

    y_positions = np.arange(y_min, y_max, step_vox)
    if len(y_positions) == 0:
        y_positions = np.array([(y_min + y_max) // 2])
    y_positions = y_positions[:num_inits]

    nx, _, nz = roi.shape
    slices = []
    for cy in y_positions:
        lo = [0, int(cy - thickness_vox // 2), 0]
        hi = [nx, int(cy + thickness_vox // 2), nz]
        slices.append(ants.crop_indices(roi, lo, hi))
    return slices


# ─── CoM-to-CoM rigid initialisation ─────────────────────────────────────────


def _vox_com(img: ants.ANTsImage) -> np.ndarray:
    """Centre-of-mass in voxel coordinates (background-filled → half-extent).
    
    Args:
        img (ants.ANTsImage): Image volume to process.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    arr = img.numpy()
    coords = np.argwhere(arr > 0)
    if len(coords) == 0:
        return np.array([s / 2.0 for s in img.shape])
    return coords.mean(axis=0)


def _phys(img: ants.ANTsImage, vc: np.ndarray) -> np.ndarray:
    """Convert voxel centroid to RAS physical coordinates.
    
    Args:
        img (ants.ANTsImage): Image volume to process.
        vc (np.ndarray): Vc value used by the operation.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    sp = np.array(img.spacing)
    d = np.array(img.direction)
    o = np.array(img.origin)
    return vc @ d * sp + o


def _init_transform_to_fixed(fixed: ants.ANTsImage, moving: ants.ANTsImage) -> str:
    """Produce a temporary ``Euler3DTransform`` that brings *moving*'s CoM onto
    
    Args:
        fixed (ants.ANTsImage): Fixed/reference image for comparison or registration.
        moving (ants.ANTsImage): Moving image to transform or compare against the fixed image.
    
    Returns:
        str: String path, label, mode, or identifier produced by the operation.
        """
    c_f = _phys(fixed, _vox_com(fixed))
    c_m = _phys(moving, _vox_com(moving))

    sp_m = np.array(moving.spacing)
    d_m = np.array(moving.direction)
    vox_delta = (c_f - c_m) @ np.linalg.inv(d_m) / sp_m

    tx = ants.create_ants_transform(
        transform_type="Euler3DTransform",
        translation=vox_delta.tolist(),
    )
    f = tempfile.NamedTemporaryFile(suffix=".mat", delete=False)
    f.close()
    ants.write_transform(tx, f.name)
    return f.name


# ─── Dice scoring ─────────────────────────────────────────────────────────────


def _compute_dice(fixed_labels: ants.ANTsImage, moving_labels: ants.ANTsImage) -> float:
    """Dice overlap via ``ants.label_overlap_measures``.
    
    Args:
        fixed_labels (ants.ANTsImage): Label data used by the operation.
        moving_labels (ants.ANTsImage): Label data used by the operation.
    
    Returns:
        float: Floating-point metric or coordinate value.
        """
    df = ants.label_overlap_measures(fixed_labels, moving_labels)
    return float(
        df.loc[df["Label"] == "All", "UnionOverlap"].values[0]
        if "UnionOverlap" in df.columns
        # Fallback for very old ANTs: 2 * TP / (A + B)
        else (
            2.0
            * np.logical_and(
                fixed_labels.numpy() > 0,
                moving_labels.numpy() > 0,
            ).sum()
            / max(
                (fixed_labels.numpy() > 0).sum() + (moving_labels.numpy() > 0).sum(), 1
            )
        )
    )


# ─── Post-mortem 3-channel label construction ────────────────────────────────


def _resample_to_match(img: ants.ANTsImage, template: ants.ANTsImage) -> ants.ANTsImage:
    """Resample *img* to have the same shape, spacing, origin and direction as
    
    Args:
        img (ants.ANTsImage): Image volume to process.
        template (ants.ANTsImage): Template value used by the operation.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    return ants.resample_image(img, reference_image=template, interp_type=0)


def _segment_postmortem_slab_gmm(
    slab_dir: Path, num_classes: int = POSTMORTEM_GMM_CLASSES
) -> ants.ANTsImage:
    """Tissue-segment a post-mortem slab with ANTs Atropos (K-Means init,
    
    Args:
        slab_dir (Path): Directory containing one post-mortem slab series.
        num_classes (int): Optional num classes value. Defaults to `POSTMORTEM_GMM_CLASSES`.
    
    Returns:
        ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
        """
    t1_img = _reorient(ants.image_read(str(slab_dir / "t1.nii.gz")))
    mask_img = _reorient(ants.image_read(str(slab_dir / "brain_mask.nii.gz")))

    gmm_seg = ants.atropos(
        a=t1_img,
        x=mask_img,
        i=f"KMeans[{num_classes}]",
        m="[0.2,1x1x1]",
    )["segmentation"]

    labels_np = gmm_seg.numpy()
    unique = np.sort(np.unique(labels_np))
    # Normalise: background = 0, then consecutively numbered.
    # Atropos processes only voxels inside the mask so its output may start at 1.
    tissue_labels = np.zeros_like(labels_np, dtype=np.float32)
    for i, lbl in enumerate(unique, start=0):
        tissue_labels[labels_np == lbl] = i

    return ants.new_image_like(gmm_seg, tissue_labels)


def _get_postmortem_tissue_labels(
    slab_dir: Path, tumour_label_img: ants.ANTsImage
) -> dict:
    """Assemble the 3-channel label representation for a post-mortem slab::
    
    Args:
        slab_dir (Path): Directory containing one post-mortem slab series.
        tumour_label_img (ants.ANTsImage): Image data used by the operation.
    
    Returns:
        dict: Dictionary containing generated metadata, paths, scores, or configuration values.
        """
    tissue_labels = _segment_postmortem_slab_gmm(slab_dir)

    tumour_channel = np.zeros_like(tissue_labels.numpy(), dtype=np.float32)
    combined_np = tissue_labels.numpy().copy()
    combined_np[tumour_channel > 0] = 3

    return {
        "tissue_labels": ants.new_image_like(tissue_labels, tissue_labels.numpy()),
        "tumour_labels": ants.new_image_like(tissue_labels, tumour_channel),
        "combined": ants.new_image_like(tissue_labels, combined_np),
    }


# ─── Slab-to-slice rigid registration ─────────────────────────────────────────


def _register_slab_to_slice(
    slab_labels: ants.ANTsImage,
    fixed_labels: ants.ANTsImage,
    moving_labels: ants.ANTsImage,
) -> dict:
    """Rigidly register *slab_labels* to *fixed_labels* and score the result with
    
    Args:
        slab_labels (ants.ANTsImage): Label data used by the operation.
        fixed_labels (ants.ANTsImage): Label data used by the operation.
        moving_labels (ants.ANTsImage): Label data used by the operation.
    
    Returns:
        dict: Dictionary containing generated metadata, paths, scores, or configuration values.
        """
    init_tx_path = _init_transform_to_fixed(fixed_labels, slab_labels)
    try:
        reg = ants.registration(
            fixed=fixed_labels,
            moving=slab_labels,
            type_of_transform="Rigid",
            initial_transform=[init_tx_path],
            metric="MeanSquares",
            metric_weight=1.0,
            reg_iterations=(1000, 500, 250, 0),
            convergence_threshold=1e-7,
            convergence_window_size=10,
            shrink_factors=(4, 3, 2, 1),
            smoothing_sigmas=(6, 4, 2, 1),
        )
        warped = reg.get("warpedmovout")
        if warped is None:
            return {}

        warped_labels = ants.apply_transforms(
            fixed=fixed_labels,
            moving=moving_labels,
            transformlist=reg["fwdtransforms"],
            interpolator="nearestNeighbor",
        )
        dice = _compute_dice(fixed_labels, warped_labels)

        return {
            "dice": dice,
            "warped_labels": warped_labels,
            "fwdtransforms": reg.get("fwdtransforms", []),
        }
    except RuntimeError:
        return {}
    finally:
        Path(init_tx_path).unlink(missing_ok=True)


# ─── Main pipeline ────────────────────────────────────────────────────────────


def run_label_based_registration(
    config_file: Path,
    patient_name: str,
) -> None:
    """Label-based registration of post-mortem slab volumes to pre-mortem full-brain.
    
    Args:
        config_file (Path): Path to the TOML configuration file.
        patient_name (str): Patient identifier to process.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    full_config = load_toml(config_file)
    config = RegistrationConfig(**full_config.get("registration", {}))
    dirs = Dirs(**full_config.get("dirs", {}))

    pre_dir = dirs.preprocessed / patient_name / "PreMortem"
    post_dir = dirs.preprocessed / patient_name / "PostMortem"
    out_dir = dirs.temp / patient_name / "registration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Pre-mortem label images ────────────────────────────────────────────
    print("Loading pre-mortem labels ...")
    pre_tissue_labels = _reorient(
        ants.image_read(str(pre_dir / "tissue_labels.nii.gz"))
    )
    pre_tumour_labels = _reorient(
        ants.image_read(str(pre_dir / "tumour_labels.nii.gz"))
    )
    seg_labels = _reorient(ants.image_read(str(pre_dir / "freesurfer_labels.nii.gz")))

    # ── 2. Tumour-side cerebrum ROI ───────────────────────────────────────────
    print("Computing tumour-side cerebrum ROI ...")
    pre_roi = _roi_to_tumour_side(pre_tissue_labels, pre_tumour_labels, seg_labels)
    tumour_bbox = _get_bbox(pre_tumour_labels)

    # ── 3. Virtual slices ─────────────────────────────────────────────────────
    print(
        f"Creating virtual slices "
        f"(thickness={SLICE_THICKNESS_MM} mm, step={SLICE_STEP_MM} mm) ..."
    )
    fixed_slices = _make_virtual_slices(
        pre_roi,
        tumour_bbox,
        num_inits=NUM_INITS,
        slice_thickness_mm=SLICE_THICKNESS_MM,
        slice_step_mm=SLICE_STEP_MM,
    )
    print(f"  {len(fixed_slices)} virtual slices created.")

    # ── 4. Register every slab × every slice ──────────────────────────────────
    pre_origin = tuple(pre_tissue_labels.origin)
    print(f"  Pre-mortem RAS origin: {pre_origin}")

    results: list[dict] = []
    slab_names = sorted(p.name for p in post_dir.iterdir() if p.is_dir())
    print(f"Found {len(slab_names)} post-mortem slab(s): {slab_names}")

    for slab_name in slab_names:
        slab_dir = post_dir / slab_name

        # 4a. Post-mortem 3-channel labels.
        post_labels = _get_postmortem_tissue_labels(slab_dir, pre_tumour_labels)
        post_tissue_l = post_labels["tissue_labels"]
        post_combined_l = post_labels["combined"]

        # 4b. Re-centre Post-mortem labels on the Pre-mortem RAS origin.
        # Without this step the two scans occupy disjoint physical space and
        # ants.registration returns an all-zero warpedmovout.
        def in_pre_space(img: ants.ANTsImage) -> ants.ANTsImage:
            """Transform a candidate point into pre-mortem image space.
            
            Args:
                img (ants.ANTsImage): Image volume to process.
            
            Returns:
                ants.ANTsImage: ANTs image containing the processed image, mask, or labels.
                """
            return ants.from_numpy(
                img.numpy(),
                spacing=img.spacing,
                origin=pre_origin,
                direction=img.direction,
            )

        post_tissue_p = in_pre_space(post_tissue_l)
        post_combined_p = in_pre_space(post_combined_l)

        # 4c-e. Register to every virtual slice and score with Dice.
        for i, fixed_slice in enumerate(fixed_slices):
            reg_result = _register_slab_to_slice(
                slab_labels=post_combined_p,
                fixed_labels=fixed_slice,
                moving_labels=post_tissue_p,
            )
            if not reg_result:
                print(f"  [{slab_name}] slice {i:03d}  FAILED")
                continue

            dice = reg_result["dice"]
            results.append(
                {
                    "slab": slab_name,
                    "slice_index": i,
                    "dice_score": round(dice, 6),
                }
            )
            print(f"  [{slab_name}] slice {i:03d}  Dice={dice:.4f}")

    if not results:
        raise RuntimeError("No successful registrations were produced.")

    # ── 5. TOP_N and manifest ────────────────────────────────────────────────
    results.sort(key=lambda r: r["dice_score"], reverse=True)
    top = results[:TOP_N]
    manifest = []

    for rank, r in enumerate(top, start=1):
        slab_name = r["slab"]
        idx = r["slice_index"]
        dice = r["dice_score"]

        manifest.append(
            {
                "rank": rank,
                "slab": slab_name,
                "slice_index": idx,
                "dice_score": dice,
            }
        )
        print(f"  Top {rank:02d}: {slab_name}  slice={idx:03d}  Dice={dice:.4f}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")
    print(f"All outputs in {out_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def get_args():
    """Parse command-line arguments for this script.
    
    Returns:
        Any: Result produced by the operation in the form described by the return annotation.
        """
    parser = argparse.ArgumentParser(
        description="Register post-mortem slab volumes to pre-mortem full-brain "
        "using label-based segmentation overlap"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent.parent / "config.toml",
        help="Path to config.toml",
    )
    parser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient name (matches preprocessed/ subfolder), e.g. IM008",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_label_based_registration(config_file=args.config, patient_name=args.patient)
