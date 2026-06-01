# GBM-REG

MRI preprocessing, segmentation, and slab-to-volume registration tools for aligning post-mortem glioblastoma slabs with in-vivo full-brain MRI.

The pipeline is configured with TOML files. It can preprocess raw NIfTI scans, segment in-vivo and ex-vivo images, initialise candidate slab positions, and run affine registration over those candidates.

This pipeline was developed with use of LLMs in a pair-programming fashion. All code has been manually reviewed and edited to align with goals.

## Repository Layout

```text
.
├── main.py                  # Runs preprocessing, segmentation, and registration
├── config.toml              # Global directories and pipeline settings
├── patients/                # Per-patient scan manifests
├── src/
│   ├── preprocess/          # Bias correction, orientation, skull stripping, normalisation
│   ├── segment/             # SynthSeg, Swin UNETR, and GMM segmentation helpers
│   ├── registration/        # Slab initialisation, masks, metrics, and affine registration
│   └── visualise/           # Plotting and overlay utilities
└── bin/                     # Small helper scripts
```

Generated data is intentionally ignored by Git. Local inputs, model weights, intermediate volumes, and registration outputs should live under directories such as `data/`, `models/`, `prep/`, `preprocessed/`, `outputs/`, `temp/`, or `render/`.

## Data Model

Each patient has a TOML file in `patients/`. The tracked examples show the expected structure:

- `PreMortem`, `PreSurgery`, and `PostSurgery` sections map MRI contrasts such as `t1`, `t1c`, `t2`, and `flair` to source NIfTI paths.
- `PostMortem` contains one section per slab series. Each slab defines `t1`, `t2`, and `flips`, where `flips` lists axes to flip during preprocessing.

Preprocessed outputs are organised as:

```text
<preprocessed>/<patient>/<timepoint>/<contrast>.nii.gz
<preprocessed>/<patient>/PostMortem/<slab>/<contrast>.nii.gz
```

Registration outputs are written below:

```text
<registered>/registration/
```

With the default `config.toml`, `<preprocessed>` is `./prep` and `<registered>` is `./outputs`.

## Installation

This project uses `uv` and Python 3.13.

```bash
uv sync
```

Some stages also depend on external neuroimaging tools or model files:

- `hd-bet` or FSL BET for skull stripping, depending on `skull_strip_algorithm`.
- FreeSurfer SynthSeg (`mri_synthseg`) for in-vivo label generation.
- Swin UNETR model weights at `./models/swinUNETR.pt` unless `config.toml` points elsewhere.
- CUDA if `segmentation.device = "cuda"`; set it to `"cpu"` for CPU inference.

## Configuration

Edit `config.toml` before running the pipeline.

Important sections:

- `[dirs]`: locations for temp files, preprocessed scans, patient manifests, and registration outputs.
- `[pipeline]`: toggles for `preprocess`, `segment`, and `register`.
- `[preprocessing]`: orientation, bias correction, skull stripping, masking, and normalisation options.
- `[segmentation]`: SynthSeg input contrast, Swin UNETR weights, device, and ex-vivo segmentation settings.
- `[registration]`: registration input type, A-P candidate spacing, candidate limits, and dry-run style switches.

For quick registration checks, set:

```toml
[registration]
max_positions = 5
skip_registration = true
```

Use `masks_only = true` when you only want masks and metadata.

## Running

Run every enabled stage for every patient manifest:

```bash
uv run python main.py --config config.toml
```

Run preprocessing for one patient:

```bash
uv run python -m src.preprocess.run --config config.toml --patient patients/IM008.toml
```

Run segmentation for one preprocessed patient:

```bash
uv run python -m src.segment.run --config config.toml --patient IM008
```

Run registration for one preprocessed patient:

```bash
uv run python -m src.registration.run --config config.toml --patient IM008
```

## Pipeline Summary

1. Preprocessing reorients images, applies N4 bias correction, registers contrasts within each timepoint, creates masks, and normalises intensities.
2. Segmentation creates in-vivo tissue and tumour labels, then segments post-mortem slabs into tissue classes.
3. Registration searches A-P slab positions, builds candidate slab volumes and masks, computes no-registration metrics, and optionally runs affine registration for intensity or segmentation inputs.
4. Visualisation utilities generate overlays and metrics plots for manual inspection.

## Notes

- Patient manifests may contain site-specific absolute paths. Keep private source data outside Git and update local manifests as needed.
- Registration and segmentation outputs can be large; keep them in ignored output directories.
- The top-level `main.py` sends success or failure notifications through `src.misc.send_ntfy` when configured.
