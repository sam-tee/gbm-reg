"""Gaussian mixture tissue segmentation wrapper."""

import argparse
from pathlib import Path

import ants


def run_gmm(
    image_path: Path, mask_path: Path, output_file: Path, num_classes: int
) -> None:
    """Segments a preprocessed image using a GMM with HMRF using AntsPy

    Args:
        image_path (Path): Filesystem path used by the operation.
        mask_path (Path): Filesystem path used by the operation.
        output_file (Path): Output file value used by the operation.
        num_classes (int): Num classes value used by the operation.

    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
    """
    image = ants.image_read(str(image_path))
    mask = ants.image_read((str(mask_path)))

    gmm_segmentation = ants.atropos(
        a=image,
        x=mask,
        m="[0.1,1x1x1]",
        i=f"KMeans[{num_classes}]",
    )["segmentation"]
    output_file.parent.mkdir(exist_ok=True, parents=True)
    ants.image_write(gmm_segmentation, str(output_file))
    print(f"Saved GMM segmentation to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment a preprocessed brain image using a Gaussian Mixture Model (GMM) via ANTsPy Atropos."
    )

    parser.add_argument(
        "-i",
        "--image",
        type=Path,
        required=True,
        help="Path to the input structural image (e.g., T1.nii.gz).",
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=Path,
        required=True,
        help="Path to the binary brain mask image.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path where the output segmentation image will be saved.",
    )

    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        default=2,
        help="Number of tissue classes for segmentation (default: 2 for GM, WM).",
    )

    args = parser.parse_args()

    run_gmm(
        image_path=args.image,
        mask_path=args.mask,
        output_file=args.output,
        num_classes=args.classes,
    )


if __name__ == "__main__":
    main()
