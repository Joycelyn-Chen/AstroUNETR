# post-process

This folder contains post-processing scripts and notebooks used to manipulate, clean, visualize, and convert mask/volume data produced by the pipeline.

## General notes
- Python 3.8+ is recommended.
- Many scripts use scientific and imaging libraries (e.g., numpy, scipy, nibabel, matplotlib, k3d, imageio). Check each script's imports and install packages as needed (for example: `pip install numpy scipy nibabel matplotlib imageio k3d`).
- Run scripts from the repository root or `post-process` directory. When in doubt run: `python3 post-process/<script>.py --help` to see available CLI options.
- Notebooks (*.ipynb) can be opened with Jupyter Lab/Notebook.

## Files
- `mag3d_SN_visualization.py`
  - Purpose: Generate 3D visualizations of magnitude signal-to-noise (S/N) from volume data.
  - Output: Interactive or static visualizations (depends on script: could use matplotlib, k3d or other 3D libs).
  - Run: `python3 post-process/mag3d_SN_visualization.py --input <volume> --out-dir <dir>`.

- `slice_da_chimney.py` and `slice_da_chimney.ipynb`
  - Purpose: Produce and/or explore slices that focus on a feature referred to informally as "chimney" (domain-specific). The `.py` is a scriptable extraction/plotting tool; the `.ipynb` provides an interactive analysis notebook.
  - Run (script): `python3 post-process/slice_da_chimney.py --input <volume> --out <dir>`.
  - Run (notebook): `jupyter lab post-process/slice_da_chimney.ipynb`.

- `temp3d_SN_visualization.py`
  - Purpose: 3D visualization of temperature S/N or similar temperature-derived fields.
  - Run: `python3 post-process/temp3d_SN_visualization.py --input <volume> --out <dir>`.

- `vel3d_SN_visualization.py`
  - Purpose: 3D visualization of velocity S/N fields.
  - Run: `python3 post-process/vel3d_SN_visualization.py --input <volume> --out <dir>`.

- `wholeCube_SN_target_k3d.py`
  - Purpose: Visualize an entire data cube's SN target using `k3d`, an interactive 3D plotting library.
  - Run: `python3 post-process/wholeCube_SN_target_k3d.py --input <cube> --out <html_or_dir>`.

- `utils.py`
  - Purpose: Collection of helper functions used by multiple scripts in this folder (I/O, mask manipulation utilities, thresholding, plotting helpers, etc.).
  - Usage: Imported by other scripts. You can open it to find reusable functions and their expected input shapes.


- For interactive visualization scripts that use `k3d` or other backends, ensure you run them in an environment that supports the chosen display (some require Jupyter or a browser).

Troubleshooting & dependencies
- If a script fails with an ImportError, install the missing package via pip. Example common packages:
  - numpy, scipy, imageio, pillow, matplotlib, nibabel, scikit-image, k3d
- If you see file path errors, double-check you are running from the repository root or supply absolute paths.