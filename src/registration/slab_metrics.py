"""Similarity metrics used by slab registration searches."""
import ants
import numpy as np


def mutual_information(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray, bins: int = 64
) -> float:
    """Compute mutual information between two image arrays inside a mask.

    Args:
        a (np.ndarray): First image array to compare.
        b (np.ndarray): Second image array to compare.
        mask (np.ndarray): Boolean mask selecting voxels used for the metric.
        bins (int): Number of histogram bins used to estimate mutual information.

    Returns:
        float: Mutual information value, or `nan` when too few masked voxels exist.
    """
    good = mask & np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 50:
        return float("nan")
    av = a[good]
    bv = b[good]
    h, _, _ = np.histogram2d(av, bv, bins=bins)
    pxy = h / np.maximum(h.sum(), 1.0)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return float((pxy[nz] * np.log(pxy[nz] / px_py[nz])).sum())


def dice_score(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Compute mean Dice for GM and WM labels inside a mask.

    Args:
        a (np.ndarray): First label array containing tissue labels.
        b (np.ndarray): Second label array containing tissue labels.
        mask (np.ndarray): Boolean mask selecting voxels used for the metric.

    Returns:
        float: Mean Dice score across labels `1` and `2`, or `nan` if no labels are present.
    """
    scores = []
    for label in (1, 2):
        aa = (a == label) & mask
        bb = (b == label) & mask
        denom = int(aa.sum() + bb.sum())
        if denom:
            scores.append(2.0 * int((aa & bb).sum()) / denom)
    return float(np.mean(scores)) if scores else float("nan")


def metrics(
    post_t2, post_seg, pre_t1c, pre_seg, roi_mask, transformlist=None
) -> tuple[float, float]:
    """Warp pre-mortem data into slab space and return MI and Dice.

    Args:
        post_t2: Fixed post-mortem T2 ANTs image defining slab space.
        post_seg: Post-mortem tissue segmentation in slab space.
        pre_t1c: Moving pre-mortem contrast-enhanced T1 ANTs image.
        pre_seg: Moving pre-mortem tissue segmentation image.
        roi_mask: Region-of-interest mask used to limit metric evaluation.
        transformlist: Optional transform file paths applied to pre-mortem images.

    Returns:
        tuple[float, float]: Mutual information and Dice score for the transformed data.
    """
    kwargs = {"transformlist": transformlist or []}
    warped_i = ants.apply_transforms(
        fixed=post_t2,
        moving=pre_t1c,
        interpolator="linear",
        **kwargs,
    )
    warped_s = ants.apply_transforms(
        fixed=post_t2,
        moving=pre_seg,
        interpolator="genericLabel",
        **kwargs,
    )
    post_mask = roi_mask.numpy() > 0
    mi = mutual_information(post_t2.numpy(), warped_i.numpy(), post_mask)
    dice = dice_score(
        post_seg.numpy().round().astype(np.uint8),
        warped_s.numpy().round().astype(np.uint8),
        post_mask,
    )
    return mi, dice
