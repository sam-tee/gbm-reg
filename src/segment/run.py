"""Command-line wrapper for segmentation workflows."""
import argparse
from pathlib import Path

from src.classes import Dirs, SegmentConfig
from src.misc import load_toml
from src.segment.pipeline import segment_exvivo, segment_invivo


def run_segmentation(config_file: Path, patient_name: str) -> None:
    """Segments all scans of a patient
    
    Args:
        config_file (Path): Path to the TOML configuration file.
        patient_name (str): Patient identifier to process.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    full_config = load_toml(config_file)
    config = SegmentConfig(**full_config.get("segmentation", {}))
    dirs = Dirs(**full_config.get("dirs", {}))
    timepoints = ["PreSurgery", "PostSurgery", "PreMortem", "PostMortem"]
    for time in timepoints:
        data_dir = dirs.preprocessed / patient_name / time
        if not data_dir.exists():
            continue
        if time == "PostMortem":
            # gets all slice directories in post mortem
            slab_names = [x.name for x in data_dir.iterdir() if x.is_dir()]
            for slab in slab_names:
                print(f"Segementing Slab: {slab}")
                segment_exvivo(patient_name, slab, config, dirs)
        else:
            scans = {
                "t1": data_dir / "t1.nii.gz",
                "t1c": data_dir / "t1c.nii.gz",
                "t2": data_dir / "t2.nii.gz",
                "flair": data_dir / "flair.nii.gz",
            }
            segment_invivo(scans, config, dirs, patient_name, time)


def get_args():
    """Parse command-line arguments for this script.
    
    Returns:
        Any: Result produced by the operation in the form described by the return annotation.
        """
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline with given config on given patient"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent.parent / "config.toml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Name of patient with preprocessed scans",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_segmentation(config_file=args.config, patient_name=args.patient)
