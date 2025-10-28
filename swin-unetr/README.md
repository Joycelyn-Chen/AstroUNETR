# [swin-unetr](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21)

This folder contains training, testing, and visualization utilities for the project. 

General notes
- Python 3.8+ recommended.
- Typical dependencies (inspect each script's imports for exact requirements): numpy, torch, torchvision, nibabel, matplotlib, SimpleITK or imageio, yaml (for config parsing), and any project-specific packages for model definitions.
- Run scripts from the repository root or set the working directory to the project root.
- To verify CLI options, run: `python3 swin-unetr/<script>.py --help` (or check the top of the script for argparse/config usage).
- For more information, checkout the original [SwinUNETR github project](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21)

## Files described

- `main.py`
  - Purpose: Entry point for experiment orchestration. Often used to parse configuration files, set up logging and device selection, and dispatch training or evaluation jobs (it may call `trainer.py` or other modules).
  - What it does: loads configuration (YAML/JSON or CLI args), prepares datasets and dataloaders, constructs the model and optimizer, sets random seeds, and either starts training or runs an evaluation depending on the configuration.
  - Run example:

```bash
python3 swin-unetr/main.py --config configs/exp.yaml
```

  - Tip: `--config` typically points to a YAML file containing dataset paths, hyperparameters, and checkpoint locations. If the script uses direct CLI flags instead, `--help` will list them.

- `nii_visualization.py`
  - Purpose: Visualize NIfTI volumes and derived outputs (slices, overlays, histogram plots). Useful for quick checks of predictions, labels, and inputs.
  - What it does: loads a `.nii` / `.nii.gz` file (or multiple), extracts slices (axial/coronal/sagittal), optionally overlays masks or predictions, and saves figures or opens interactive displays.
  - Run example:

```bash
python3 swin-unetr/nii_visualization.py --input /path/to/volume.nii.gz --out-dir results/figs --slice 50 --axis axial
```

  - Tip: If the script supports multiple overlays, provide both image and mask paths. Some visualization scripts accept a colormap, intensity windowing, or output DPI flags.

- `nii-astro-visualization.py`
  - Purpose: Domain-specific NIfTI visualization tailored for astrophysical imaging conventions used in this project (color scales, SN plotting, specialized overlays, etc.).
  - What it does: similar to `nii_visualization.py` but with astrophysics-specific plotting defaults (e.g., log-scaling, custom colorbars, S/N annotations). Use this when you want consistent figures for paper/analysis.
  - Run example:

```bash
python3 swin-unetr/nii-astro-visualization.py --input /path/to/volume.nii --mask /path/to/mask.nii --out-dir results/astro_figs --slice 40
```

- `test.py`
  - Purpose: Run evaluation / inference using a saved model checkpoint on a test dataset.
  - What it does: loads a model checkpoint, prepares the test dataset/dataloader, performs forward passes to compute predictions, writes predicted masks or metrics to disk, and prints/saves evaluation results (e.g., Dice, IoU).
  - Run example:

```bash
python3 swin-unetr/test.py --data-dir /path/to/test_data --checkpoint /path/to/checkpoint.pth --out-dir results/test_preds --batch-size 2
```

  - Tip: Use `--device cuda` or `--device cpu` as supported. If your checkpoint was trained with DataParallel, ensure `test.py` can map keys or load appropriately.

- `trainer.py`
  - Purpose: Contains the training loop, optimization step, checkpointing, and (optionally) validation logic used by the project.
  - What it does: given model, optimizer, dataloaders and configuration, runs epoch loops, computes losses and metrics, saves periodic checkpoints, and emits logs for monitoring.
  - Run example (direct invocation if supported):

```bash
python3 swin-unetr/trainer.py --config configs/train.yaml
```

  - Tip: Often `main.py` wraps `trainer.py`. Call `trainer.py` directly for quick experiments, or call `main.py` if you prefer higher-level orchestration and automatic logging.
