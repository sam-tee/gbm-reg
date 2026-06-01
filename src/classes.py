"""Configuration models used by preprocessing, segmentation, and registration pipelines."""
from pathlib import Path

from pydantic import BaseModel


class PreprocessConfig(BaseModel):
    """Pydantic model describing preprocessing options.
    
    Attributes:
        orientation (str): Configuration or state value used by the pipeline.
        bias_shrink_factor (int): Configuration or state value used by the pipeline.
        skull_strip_algorithm (str): Configuration or state value used by the pipeline.
        hd_bet_flags (list[str] | None): Configuration or state value used by the pipeline.
        fsl_bet_threshold (float): Configuration or state value used by the pipeline.
        tissue_mask_algorithm (str): Configuration or state value used by the pipeline.
        normalisation_method (str): Configuration or state value used by the pipeline.
        min_intensity (int): Configuration or state value used by the pipeline.
        max_intensity (int): Configuration or state value used by the pipeline.
        """
    orientation: str = "RAS"
    bias_shrink_factor: int = 2
    skull_strip_algorithm: str = "hd-bet"
    hd_bet_flags: list[str] | None = None
    fsl_bet_threshold: float = 0.5
    tissue_mask_algorithm: str = "gmm"
    normalisation_method: str = "z-score"
    min_intensity: int = 0
    max_intensity: int = 255


class Dirs(BaseModel):
    """Pydantic model containing key pipeline directory paths.
    
    Attributes:
        temp (Path): Configuration or state value used by the pipeline.
        preprocessed (Path): Configuration or state value used by the pipeline.
        patients (Path): Configuration or state value used by the pipeline.
        registered (Path): Configuration or state value used by the pipeline.
        """
    temp: Path
    preprocessed: Path
    patients: Path
    registered: Path = Path("./outputs")


class PostMortemSeries(BaseModel):
    """Pydantic model for one post-mortem slab series.
    
    Attributes:
        t1 (Path): Configuration or state value used by the pipeline.
        t2 (Path): Configuration or state value used by the pipeline.
        flips (tuple): Configuration or state value used by the pipeline.
        """
    t1: Path
    t2: Path
    flips: tuple


class Patient(BaseModel):
    """Pydantic model describing one patient and associated scan series.
    
    Attributes:
        PreMortem (dict[str, str]): Configuration or state value used by the pipeline.
        PostMortem (dict[str, PostMortemSeries]): Configuration or state value used by the pipeline.
        PreSurgery (dict[str, str] | None): Configuration or state value used by the pipeline.
        PostSurgery (dict[str, str] | None): Configuration or state value used by the pipeline.
        """
    PreMortem: dict[str, str]
    PostMortem: dict[str, PostMortemSeries]
    PreSurgery: dict[str, str] | None = None
    PostSurgery: dict[str, str] | None = None


class SegmentConfig(BaseModel):
    """Pydantic model describing segmentation model and label settings.
    
    Attributes:
        synthseg_scan (str): Configuration or state value used by the pipeline.
        synthseg_flags (list[str] | None): Configuration or state value used by the pipeline.
        swin_unetr_model_path (Path | None): Configuration or state value used by the pipeline.
        device (str): Configuration or state value used by the pipeline.
        exvivo_label_algorithm (str): Configuration or state value used by the pipeline.
        gmm_classes (int): Configuration or state value used by the pipeline.
        medsam2_checkpoint (Path): Configuration or state value used by MedSAM2.
        medsam2_model_cfg (str): Configuration or state value used by MedSAM2.
        medsam2_repo_path (Path | None): Optional local MedSAM2 source checkout.
        medsam2_prompt_scan (str): Ex-vivo contrast used for MedSAM2 inference.
        medsam2_slice_axis (int): Volume axis treated as MedSAM2 propagation depth.
        medsam2_bbox_margin (int): Margin added around the automatic box prompt.
        medsam2_image_size (int): Input slice size used by the MedSAM2 model.
        medsam2_tumour_labels (bool): Whether to create post-mortem tumour labels.
        medsam2_tumour_label (int): Label value used for post-mortem tumour ROI.
        medsam2_tumour_prompt_percentile (float): T2 percentile for tumour prompt.
        medsam2_tumour_prompt_min_fraction (float): Minimum tumour prompt fraction.
        """
    synthseg_scan: str = "t1"
    synthseg_flags: list[str] | None = None
    swin_unetr_model_path: Path | None = None
    device: str = "cuda"
    exvivo_label_algorithm: str = "gmm"
    gmm_classes: int = 3
    medsam2_checkpoint: Path = Path("./models/MedSAM2_latest.pt")
    medsam2_model_cfg: str = "configs/sam2.1_hiera_t512.yaml"
    medsam2_repo_path: Path | None = None
    medsam2_prompt_scan: str = "t2"
    medsam2_slice_axis: int = 2
    medsam2_bbox_margin: int = 8
    medsam2_image_size: int = 512
    medsam2_tumour_labels: bool = True
    medsam2_tumour_label: int = 2
    medsam2_tumour_prompt_percentile: float = 90.0
    medsam2_tumour_prompt_min_fraction: float = 0.002


class RegistrationConfig(BaseModel):
    """Pydantic model describing registration inputs and search settings.
    
    Attributes:
        image_type (str): Configuration or state value used by the pipeline.
        ap_step_mm (float): Configuration or state value used by the pipeline.
        max_positions (int | None): Configuration or state value used by the pipeline.
        skip_registration (bool): Configuration or state value used by the pipeline.
        masks_only (bool): Configuration or state value used by the pipeline.
        """
    image_type: str = "segmentation"
    ap_step_mm: float = 5.0
    max_positions: int | None = None
    skip_registration: bool = False
    masks_only: bool = False
