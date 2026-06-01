"""Command-line wrapper for running preprocessing from config files."""
import argparse
from pathlib import Path

from src.classes import Dirs, Patient, PreprocessConfig
from src.misc import load_toml
from src.preprocess.pipeline import preprocess_exvivo, preprocess_invivo


def run_preprocessing(config_file: Path, patient_file: Path) -> None:
    """Run preprocessing for a patient file and global config.
    
    Args:
        config_file (Path): Path to the TOML configuration file.
        patient_file (Path): Path to the patient TOML file.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    full_config = load_toml(config_file)
    config = PreprocessConfig(**full_config.get("preprocessing", {}))
    dirs = Dirs(**full_config.get("dirs", {}))
    patient = Patient(**load_toml(patient_file))
    patient_name = patient_file.name[:-5]
    for field_name in Patient.model_fields:
        scans = getattr(patient, field_name)
        if scans is None:
            print("No scans, skipping")
            continue
        print(f"--- Preprocessing {field_name} scans ---")
        if field_name == "PostMortem":
            preprocess_exvivo(scans, config, dirs, patient_name, field_name)
        else:
            preprocess_invivo(scans, config, dirs, patient_name, field_name)


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
        type=Path,
        required=True,
        help="Path to the patient toml file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_preprocessing(config_file=args.config, patient_file=args.patient)
