"""I/O helpers for slab registration transforms and ANTs images."""
import shutil
from pathlib import Path

import ants


def load_ants(path: Path, pixeltype: str = "float"):
    """Load an image from disk as an ANTs image.

    Args:
        path (Path): Filesystem path to the image to read.
        pixeltype (str): ANTs pixel type to request when loading the image.

    Returns:
        ants.ANTsImage: Loaded ANTs image with the requested pixel type.
    """
    return ants.image_read(str(path), pixeltype=pixeltype)


def copy_transforms(
    transform_paths: list[str], dest_dir: Path, prefix: str
) -> list[str]:
    """Copy generated transform files into the result directory.

    Args:
        transform_paths (list[str]): Transform file paths produced by registration.
        dest_dir (Path): Directory where copied transforms should be stored.
        prefix (str): Filename prefix applied to copied transform files.

    Returns:
        list[str]: Paths to the copied transform files as strings.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for i, src in enumerate(transform_paths):
        src_path = Path(src)
        if src_path.parent.resolve() == dest_dir.resolve():
            copied.append(str(src_path))
            continue
        dest = dest_dir / f"{prefix}_{i}_{src_path.name}"
        shutil.copy2(src_path, dest)
        copied.append(str(dest))
    return copied
