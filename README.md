# AstroUNETR
[![DOI](https://zenodo.org/badge/1085115056.svg)](https://doi.org/10.5281/zenodo.17468097)

This repository contains code and utilities developed in support of the paper submission entitled:

"Segmenting Superbubbles in a simulated Multiphase Interstellar Medium using Computer Vision"

## Summary
This work explores automated 3D superbubble segmentation in a simulation environment. The goal is to streamline superbubble analysis by providing methods that accurately track and robustly segment bubble shapes from volumetric simulation outputs. The codebase collects data conversion tools, model training and evaluation scripts, visualization utilities, and post-processing helpers used during development and experiments.

## Repository layout
Below is a short description of each top-level folder and pointers to more detailed READMEs where available.

- `analysis/`
  - Lightweight analysis scripts (plotting, energy/time traces) used for exploratory checks. See files in the folder for specific utilities.

- `data/` — data conversion and HDF5 utilities
  - Tools for converting and inspecting dataset formats (HDF5, PNG/JPG, NIfTI) and generating masks. See `data/README.md` for details and examples.

- `evaluation/`
  - Scripts to evaluate model outputs against ground truth (metrics, formatting helpers). Check the folder for evaluation scripts like `evaluate-nii.py`.

- `graphs/` — plotting and visualization
  - Utilities to generate plots and figures for analysis and paper figures. See `graphs/README.md` for per-script usage and examples.

- `post-process/` — post-processing & visualization helpers
  - Scripts for cleaning, merging, and visualizing masks and volumes (3D viewers, k3d exporters). See `post-process/README.md` for details.

- `sam2/` — instance segmentation / SAM-based experiments
  - Demo and utility scripts for instance segmentation and tracking (point-prompt based segmentation, video inference). See `sam2/README.md`.

- `segresnet/` — ResNet-based segmentation experiments
  - Training and testing scripts for a ResNet-style segmentation baseline. See `segresnet/README.md`.

- `swin-unetr/` — Swin-UNETR model code and helpers
  - Training, testing, and NIfTI visualization code for Swin-UNETR experiments. See `swin-unetr/README.md` for the main scripts and run examples.

- `unet-3d/` — alternative UNet3D experiments
  - Older or parallel experiments for 3D U-Net training and testing. Inspect the folder for `train.py` / `test.py` scripts.

Getting started
- Read the README in the subfolder that matches the task you want to run (data conversion, training, evaluation, or visualization). Many scripts support `--help`; 

License
- See `LICENSE` at the repository root.
