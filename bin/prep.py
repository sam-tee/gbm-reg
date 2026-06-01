"""Utilities for preparing raw imaging archives for conversion to NIfTI."""
import argparse
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path


def recursive_unzip(target_dir: Path):
    """Recursives extracts all .zip, .tar, .tar.gz files
    
    Args:
        target_dir (Path): Directory to search or prepare.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    exts = ["*.zip", "*.tar", "*.tar.gz"]
    for ext in exts:
        for archive in target_dir.rglob(ext):
            extract_to = archive.parent / archive.stem
            extract_to.mkdir(exist_ok=True, parents=True)
            try:
                if archive.suffix == ".zip":
                    with zipfile.ZipFile(archive, "r") as zip_ref:
                        zip_ref.extractall(extract_to)
                elif ".tar" in archive.suffixes:
                    with tarfile.open(archive, "r:*") as tar_ref:
                        tar_ref.extractall(extract_to)
                archive.unlink()
            except FileNotFoundError:
                continue


def dicom2nifti(dicom_dir: Path, nifti_dir: Path):
    """Convert a DICOM directory to compressed NIfTI files.
    
    Args:
        dicom_dir (Path): Directory containing DICOM inputs.
        nifti_dir (Path): Directory where NIfTI outputs are written.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    nifti_dir.mkdir(exist_ok=True, parents=True)
    for f in dicom_dir.rglob("*.nii*"):
        relative_path = f.relative_to(dicom_dir)
        dest_f = nifti_dir / relative_path
        if dest_f.exists():
            continue
        dest_f.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(f, dest_f)
    for patient_dir in dicom_dir.iterdir():
        if not patient_dir.is_dir():
            continue
        for time_dir in patient_dir.iterdir():
            if not time_dir.is_dir():
                continue
            time_name = time_dir.name.replace(" ", "")
            patient_name = patient_dir.name.replace(" ", "")
            output_dir = nifti_dir / patient_name / time_name
            output_dir.mkdir(exist_ok=True, parents=True)
            cmd = [
                "dcm2niix",
                "-o",
                str(output_dir),
                "-f",
                "%p___%q___%z___%d",
                "-z",
                "y",
                "-b",
                "n",
                str(time_dir),
            ]
            subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    recursive_unzip(args.dir)
    dicom2nifti(args.dir, args.output)
