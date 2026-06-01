"""SynthSeg execution and FreeSurfer label remapping helpers."""
import subprocess
from pathlib import Path

import ants

from src.segment.labels import remap_freesurfer_labels


def run_synthseg(
    image_path: Path, output_dir: Path, flags: list[str] | None = None
) -> None:
    """Run SynthSeg on an input image and write segmentation outputs.

    Args:
        image_path (Path): Path to the image that SynthSeg should segment.
        output_dir (Path): Directory where FreeSurfer and tissue label outputs are written.
        flags (list[str] | None): Optional command-line flags passed through to SynthSeg.

    Returns:
        None: This function returns `None`; segmentation outputs are written to disk.
    """
    cmd = [
        "mri_synthseg",
        "--i",
        image_path,
        "--o",
        output_dir / "freesurfer_labels.nii.gz",
    ]
    if flags is not None:
        cmd.extend(flags)
    subprocess.run(cmd, capture_output=True)
    fs_labels = ants.image_read(str(output_dir / "freesurfer_labels.nii.gz"))
    tissue_labels_image = remap_freesurfer_labels(fs_labels)
    ants.image_write(tissue_labels_image, str(output_dir / "tissue_labels.nii.gz"))
