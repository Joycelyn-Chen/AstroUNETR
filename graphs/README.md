# Graphs

This folder contains plotting and visualization utilities used to produce figures from model outputs, volumes, and masks. 

- Run scripts from the repository root or from the `graphs` directory. Example: `python3 graphs/<script>.py --help` to see available CLI options.
- Common dependencies: numpy, matplotlib, seaborn, imageio, nibabel, h5py, scikit-image. Install as needed: `pip install numpy matplotlib seaborn imageio nibabel h5py scikit-image`.
- Notebooks (`.ipynb`) can be opened with Jupyter Lab/Notebook.

## Files

- `hdf52sliceplot.py`
  - Purpose: Create slice plots from data stored in HDF5 files. Useful for quick inspection of 2D slices across a volume or time-series.
  - Typical behavior: loads datasets from an HDF5 file, extracts slices (axis configurable), and saves PNGs or a multi-page figure.
  - Run: `python3 graphs/hdf52sliceplot.py --input <data.hdf5 folder> --dataset </path/to/dset> --out-dir <figs>`.

- `hist-of-temp.py` and `hist-of-temp.ipynb`
  - Purpose: Produce histograms of temperature values (or another scalar field) and provide an interactive notebook for exploration.
  - Run (script): `python3 graphs/hist-of-temp.py --input <volume.nii|.h5|.npy> --out hist_temp.png`.
  - Run (notebook): `jupyter lab graphs/hist-of-temp.ipynb`.

- `performance-box-plot.py`
  - Purpose: Generate box plots comparing performance across experimental conditions, models, or modalities.
  - Typical behavior: reads per-sample performance metrics from directories or CSVs and plots grouped box plots.
  - Run: `python3 graphs/performance-box-plot.py --root_dir <predictions_root> --gt_dir <ground_truth_root> --out perf_box.png`.

Troubleshooting
- Missing imports? Install the missing package with pip.
- File-format errors? Confirm input types (e.g., `.h5`, `.nii`, `.npy`, `.png`) match what the script expects.
