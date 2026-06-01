"""
Taken from https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/BRATS21/test.py
with some changes made to fit codebase
"""

from functools import partial
from pathlib import Path

import ants
import numpy as np
import torch
from monai import data, transforms
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR


def get_loader(scans: dict[str, Path]) -> data.DataLoader:
    """Create a MONAI data loader for Swin UNETR inference inputs.
    
    Args:
        scans (dict[str, Path]): Mapping of scan names to input image paths.
    
    Returns:
        data.DataLoader: Result produced by the operation in the form described by the return annotation.
        """
    image = np.stack(
        [
            ants.image_read(str(scans["flair"])).numpy(),
            ants.image_read(str(scans["t1c"])).numpy(),
            ants.image_read(str(scans["t1"])).numpy(),
            ants.image_read(str(scans["t2"])).numpy(),
        ],
        axis=0,
    ).astype(np.float32)
    test_files = [{"image": image, "reference": str(scans["t1c"])}]
    test_transform = transforms.Compose(
        [
            transforms.ToTensord(keys=["image"]),
        ]
    )
    test_ds = data.Dataset(data=test_files, transform=test_transform)

    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return test_loader


def init_model(device: torch.device, model_path: Path) -> SwinUNETR:
    """Initialise a Swin UNETR model and load checkpoint weights.
    
    Args:
        device (torch.device): Torch device on which inference should run.
        model_path (Path): Path to trained model weights.
    
    Returns:
        SwinUNETR: Result produced by the operation in the form described by the return annotation.
        """
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)["state_dict"]
    )
    model.eval()
    model.to(device)
    return model


def make_physical_brats_labels(seg: np.ndarray) -> np.ndarray:
    """Uses
    
    Args:
        seg (np.ndarray): Seg value used by the operation.
    
    Returns:
        np.ndarray: NumPy array containing the processed image, mask, label, or metric data.
        """
    seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]), dtype=np.int8)
    seg_out[seg[1] == 1] = 2
    seg_out[seg[0] == 1] = 1
    seg_out[seg[2] == 1] = 4

    edema = seg[1] == 1
    enhancing = (seg[2] == 1) & edema
    necrotic = (seg[0] == 1) & enhancing

    seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]), dtype=np.int8)
    seg_out[edema] = 2
    seg_out[enhancing] = 4
    seg_out[necrotic] = 1
    return seg_out


def run_swin_unetr(
    scans: dict[str, Path],
    model_path: Path,
    out_dir: Path,
    device: str,
):
    """Run Swin UNETR tumour segmentation for a set of MRI scans.
    
    Args:
        scans (dict[str, Path]): Mapping of scan names to input image paths.
        model_path (Path): Path to trained model weights.
        out_dir (Path): Directory path used by the operation.
        device (str): Torch device on which inference should run.
    
    Returns:
        Any: Result produced by the operation in the form described by the return annotation.
        """
    if device == "cuda" and torch.cuda.is_available():
        torch_device = torch.device(device)
    elif device == "cuda":
        print("Error: CUDA initialisation failed - falling back to cpu")
        torch_device = torch.device("cpu")
    model = init_model(torch_device, model_path)
    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[128, 128, 128],
        sw_batch_size=1,
        predictor=model,
        overlap=0.6,
    )
    test_loader = get_loader(scans)
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(torch_device)
            reference = ants.image_read(batch["reference"][0])
            prob = torch.sigmoid(model_inferer_test(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)

            seg_out = np.zeros(
                (seg.shape[1], seg.shape[2], seg.shape[3]), dtype=np.int8
            )
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4

            out_path = out_dir / "tumour_labels.nii.gz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            seg_image = ants.new_image_like(reference, seg_out.astype(np.uint8))
            ants.image_write(seg_image, str(out_path))
            print(f"Saved segmentation to {out_path}")
        print("Finished inference!")
    return seg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SwinUNETR on BRATS21 data")
    parser.add_argument(
        "--scans",
        type=Path,
        required=True,
        help="Path to directory containing the 4 scans (flair, t1, t1c, t2)",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to the trained model weights",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory to save the output segmentation to",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )
    args = parser.parse_args()

    seg = run_swin_unetr(
        scans={
            "flair": args.scans / "flair.nii.gz",
            "t1": args.scans / "t1.nii.gz",
            "t1c": args.scans / "t1c.nii.gz",
            "t2": args.scans / "t2.nii.gz",
        },
        model_path=args.model_path,
        out_dir=args.out_dir,
        device=args.device,
    )
    np.save(args.out_dir / "segmentation.npy", seg)
