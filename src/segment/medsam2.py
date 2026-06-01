"""Automated MedSAM2 mask generation for post-mortem slab volumes."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage


ImageLike = np.ndarray


def _largest_connected_component(mask: ImageLike) -> ImageLike:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    labels, n_labels = ndimage.label(mask, structure=structure)
    if n_labels == 0:
        return mask.astype(bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(counts.argmax())


def _normalise_to_uint8(data: ImageLike, mask: ImageLike) -> ImageLike:
    values = data[mask > 0]
    if values.size == 0:
        values = data[np.isfinite(data)]
    if values.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)

    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)

    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return np.uint8(np.round(scaled * 255.0))


def _move_axis_to_depth(data: ImageLike, axis: int) -> ImageLike:
    return np.moveaxis(data, axis, 0)


def _restore_depth_axis(data: ImageLike, axis: int) -> ImageLike:
    return np.moveaxis(data, 0, axis)


def _resize_depth_slices(volume: ImageLike, image_size: int) -> ImageLike:
    depth, height, width = volume.shape
    zoom = (1.0, image_size / height, image_size / width)
    resized = ndimage.zoom(volume, zoom, order=1)
    rgb = np.repeat(resized[:, None, :, :], 3, axis=1)
    return rgb.astype(np.float32) / 255.0


def _key_slice_and_box(mask_dhw: ImageLike, margin: int) -> tuple[int, np.ndarray]:
    slice_areas = mask_dhw.reshape(mask_dhw.shape[0], -1).sum(axis=1)
    if int(slice_areas.max()) == 0:
        raise ValueError("Cannot create MedSAM2 prompt: prompt mask is empty.")

    key_slice = int(slice_areas.argmax())
    y_idx, x_idx = np.where(mask_dhw[key_slice] > 0)
    height, width = mask_dhw.shape[1:]
    x_min = max(0, int(x_idx.min()) - margin)
    x_max = min(width - 1, int(x_idx.max()) + margin)
    y_min = max(0, int(y_idx.min()) - margin)
    y_max = min(height - 1, int(y_idx.max()) + margin)
    return key_slice, np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def _automated_tumour_prompt(
    data: ImageLike,
    tissue_mask: ImageLike,
    percentile: float,
    min_fraction: float,
) -> ImageLike:
    tissue = tissue_mask > 0
    values = data[tissue & np.isfinite(data)]
    if values.size == 0:
        raise ValueError("Cannot create tumour prompt: tissue mask is empty.")

    threshold = np.percentile(values, percentile)
    prompt = (data >= threshold) & tissue
    min_voxels = max(1, int(tissue.sum() * min_fraction))
    while prompt.sum() < min_voxels and percentile > 50.0:
        percentile -= 5.0
        threshold = np.percentile(values, percentile)
        prompt = (data >= threshold) & tissue

    prompt = ndimage.binary_opening(prompt, iterations=1)
    prompt = ndimage.binary_closing(prompt, iterations=1)
    if not np.any(prompt):
        prompt = (data >= threshold) & tissue
    return _largest_connected_component(prompt)


def _load_predictor(repo_path: Path | None, model_cfg: str, checkpoint: Path) -> Any:
    if repo_path is not None:
        sys.path.insert(0, str(repo_path.expanduser().resolve()))
    try:
        from sam2.build_sam import build_sam2_video_predictor_npz
    except ImportError as exc:
        raise RuntimeError(
            "MedSAM2 is not importable. Install bowang-lab/MedSAM2 in the active "
            "environment or set segmentation.medsam2_repo_path to a local clone."
        ) from exc
    return build_sam2_video_predictor_npz(model_cfg, str(checkpoint))


def run_medsam2_mask(
    image_path: Path,
    prompt_mask_path: Path,
    output_path: Path,
    checkpoint: Path,
    model_cfg: str,
    repo_path: Path | None = None,
    device: str = "cuda",
    slice_axis: int = 2,
    bbox_margin: int = 8,
    image_size: int = 512,
) -> Path:
    """Segment a slab automatically by prompting MedSAM2 from an existing mask.

    The post-mortem preprocessing step already creates a rough tissue mask. This
    function turns that mask into a key-slice bounding box prompt, lets MedSAM2
    propagate the object through the slab volume, and writes a binary NIfTI mask.
    """
    img = nib.load(image_path)
    prompt_img = nib.load(prompt_mask_path)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    prompt_mask = np.asanyarray(prompt_img.dataobj) > 0
    prompt_mask = _largest_connected_component(prompt_mask)

    if data.shape != prompt_mask.shape:
        raise ValueError(
            f"MedSAM2 image and prompt mask shapes differ: {data.shape} vs "
            f"{prompt_mask.shape}"
        )
    if slice_axis < 0:
        slice_axis += data.ndim
    if slice_axis < 0 or slice_axis >= data.ndim:
        raise ValueError(f"slice_axis must be in [0, {data.ndim - 1}], got {slice_axis}")

    data_u8 = _normalise_to_uint8(data, prompt_mask)
    data_dhw = _move_axis_to_depth(data_u8, slice_axis)
    prompt_dhw = _move_axis_to_depth(prompt_mask.astype(np.uint8), slice_axis)
    key_slice, box = _key_slice_and_box(prompt_dhw, bbox_margin)

    predictor = _load_predictor(repo_path, model_cfg, checkpoint)
    tensor = torch.from_numpy(_resize_depth_slices(data_dhw, image_size))
    tensor = tensor.to(device)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=device)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=device)
    tensor = (tensor - mean[:, None, None]) / std[:, None, None]

    video_height, video_width = data_dhw.shape[1:]
    mask_dhw = np.zeros(data_dhw.shape, dtype=np.uint8)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )

    with torch.inference_mode(), autocast:
        state = predictor.init_state(tensor, video_height, video_width)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=key_slice,
            obj_id=1,
            box=box,
        )
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            mask_dhw[frame_idx, (mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(state)

        state = predictor.init_state(tensor, video_height, video_width)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=key_slice,
            obj_id=1,
            box=box,
        )
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
            state, reverse=True
        ):
            mask_dhw[frame_idx, (mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(state)

    mask = _restore_depth_axis(mask_dhw, slice_axis).astype(bool)
    if np.any(mask):
        mask = _largest_connected_component(mask)
    else:
        raise RuntimeError("MedSAM2 returned an empty slab mask.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(mask.astype(np.uint8), img.affine, img.header)
    nib.save(out_img, output_path)
    print(f"Saved MedSAM2 mask to {output_path}")
    return output_path


def run_medsam2_tumour_labels(
    image_path: Path,
    tissue_mask_path: Path,
    output_path: Path,
    checkpoint: Path,
    model_cfg: str,
    repo_path: Path | None = None,
    device: str = "cuda",
    slice_axis: int = 2,
    bbox_margin: int = 8,
    image_size: int = 512,
    tumour_label: int = 2,
    tumour_prompt_percentile: float = 90.0,
    tumour_prompt_min_fraction: float = 0.002,
) -> Path:
    """Write automated post-mortem tumour labels using MedSAM2 refinement.

    The tumour prompt is estimated from high T2 intensity within the slab tissue
    mask. MedSAM2 then refines and propagates that candidate region through the
    volume. The output follows the in-vivo tumour label convention and uses
    label 2 by default, because downstream registration treats label 2 as the
    oedema/tumour ROI.
    """
    img = nib.load(image_path)
    tissue_img = nib.load(tissue_mask_path)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    tissue_mask = np.asanyarray(tissue_img.dataobj) > 0
    prompt = _automated_tumour_prompt(
        data,
        tissue_mask,
        percentile=tumour_prompt_percentile,
        min_fraction=tumour_prompt_min_fraction,
    )

    prompt_path = output_path.with_name(output_path.name.replace(".nii.gz", "_prompt.nii.gz"))
    nib.save(
        nib.Nifti1Image(prompt.astype(np.uint8), img.affine, img.header),
        prompt_path,
    )
    mask_path = output_path.with_name(output_path.name.replace(".nii.gz", "_mask.nii.gz"))
    run_medsam2_mask(
        image_path=image_path,
        prompt_mask_path=prompt_path,
        output_path=mask_path,
        checkpoint=checkpoint,
        model_cfg=model_cfg,
        repo_path=repo_path,
        device=device,
        slice_axis=slice_axis,
        bbox_margin=bbox_margin,
        image_size=image_size,
    )

    mask_img = nib.load(mask_path)
    mask = np.asanyarray(mask_img.dataobj) > 0
    labels = np.zeros(mask.shape, dtype=np.uint8)
    labels[mask] = tumour_label
    nib.save(nib.Nifti1Image(labels, mask_img.affine, mask_img.header), output_path)
    print(f"Saved MedSAM2 tumour labels to {output_path}")
    return output_path
