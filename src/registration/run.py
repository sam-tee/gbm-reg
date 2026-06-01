"""Command-line wrapper for registration workflows."""
import argparse
from pathlib import Path

from src.classes import Dirs, RegistrationConfig
from src.misc import load_toml
from src.registration.slab_pipeline import Paths, run


def run_registration(config_file: Path, patient_name: str | None = None) -> None:
    """Run the slab-to-pre-mortem registration pipeline.
    
    Args:
        config_file (Path): Path to the TOML configuration file.
        patient_name (str | None): Patient identifier to process.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    full_config = load_toml(config_file)
    config = RegistrationConfig(**full_config.get("registration", {}))
    dirs = Dirs(**full_config.get("dirs", {}))

    patients = [patient_name] if patient_name else None
    run(
        Paths(dirs.preprocessed, dirs.registered / "registration"),
        patients=patients,
        max_positions=config.max_positions,
        skip_registration=config.skip_registration,
        ap_step_mm=config.ap_step_mm,
        masks_only=config.masks_only,
    )


def get_args():
    """Parse command-line arguments for this script.
    
    Returns:
        Any: Result produced by the operation in the form described by the return annotation.
        """
    parser = argparse.ArgumentParser(
        description="Run slab-to-pre-mortem registration for one or more patients"
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
        default=None,
        help="Optional patient name matching a preprocessed subfolder, e.g. IM008",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_registration(config_file=args.config, patient_name=args.patient)
