# Data

These utilities help inspect and convert HDF5-stored volumes/datasets into 2D slices, projections, and image formats for quick inspection or downstream processing. 

Common requirements
- Python 3.8+
- Common Python packages used: h5py, numpy, imageio, matplotlib, pillow. Install as needed, for example:

```bash
pip install h5py numpy imageio matplotlib pillow
```

## Files

- `hdf52projections.py`
  - Purpose: Compute and save 2D projections (e.g., maximum intensity projection, mean projection, sum projection) from a 3D HDF5 volume along a chosen axis.
  - What it does: loads the specified dataset from an HDF5 file, computes the chosen projection along axis (0,1,2), optionally applies simple normalization or clipping, and writes the resulting 2D image(s) as PNG/JPEG.
  - Example usage:

```bash
python3 data/hdf52projections.py --input /path/to/data.hdf5 --dataset /volume --axis 0 --method max --out /out/dir/proj_axis0.png
```

- `hdf52sliceplots.py`
  - Purpose: Extract 2D slices from a 3D HDF5 volume and create visual plots (single images, grids, or multi-page figures) for slice-by-slice inspection.
  - What it does: opens the HDF5 file, selects slices at given indices or an evenly spaced set across an axis, and saves PNGs or a combined figure showing multiple slices.
  - Example usage:

```bash
python3 data/hdf52sliceplots.py --input /path/to/data.h5 --dataset /volume --axis 2 --slices 10,20,30 --out-dir /out/slices
```

- `hdf5tojpg.py`
  - Purpose: Convert 2D datasets or extracted projections/slices stored in HDF5 into JPG/PNG image files. Useful for quickly converting arrays into common image formats that many viewers accept.
  - What it does: reads a dataset or selection from an HDF5 file, optionally rescales intensity to 8-bit, and writes one or more JPG/PNG images.
  - Example usage:

```bash
python3 data/hdf5tojpg.py --input /path/to/data.h5 --dataset /volume/slice_000 --out /out/img000.jpg
```
