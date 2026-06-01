"""Command-line entry point for running preprocessing, segmentation, and registration stages."""
import argparse
from pathlib import Path

from src.misc import load_toml, send_ntfy
from src.preprocess.run import run_preprocessing
from src.registration.run import run_registration
from src.segment.run import run_segmentation


def get_args():
    """Parse command-line arguments for this script.
    
    Returns:
        Any: Result produced by the operation in the form described by the return annotation.
        """
    parser = argparse.ArgumentParser(
        description="Run complete pipeline for slab to full brain registration"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.toml",
        help="Path to the config file",
    )
    return parser.parse_args()


def main():
    """Run the script entry point.
    
    Returns:
        None: This function returns `None`; outputs are written to disk, plotted, logged, or applied through side effects.
        """
    args = get_args()
    config_file = args.config
    full_config = load_toml(config_file)
    directories = full_config.get("dirs", {})
    pipeline_config = full_config.get("pipeline", {})
    patient_dir = Path(directories.get("patients", {}))
    for patient_toml in patient_dir.glob("*.toml"):
        if pipeline_config.get("preprocess", True):
            run_preprocessing(config_file, patient_toml)
        patient_name = patient_toml.name[:-5]
        if pipeline_config.get("segment", True):
            run_segmentation(config_file, patient_name)
        if pipeline_config.get("register", True):
            run_registration(config_file, patient_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_type = type(e).__name__
        error_message = f"Error Type: {error_type}\nError: {e}"

        send_ntfy(
            message=error_message,
            title="Pipeline Failed",
            priority="high",
        )
        raise
    else:
        send_ntfy(message="Pipeline completed successfully", title="Pipeline Success")
