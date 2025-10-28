import os
import yt
import numpy as np
import k3d
from k3d import matplotlib_color_maps
import argparse

from utils import *

DEBUG = True
hdf5_prefix = 'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0'

def get_velocity_data(obj, x_range, y_range, z_range):
    """
    Retrieve velx, vely, velz, density, and temperature in a single function.
    """
    velx = obj["flash", "velx"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('km/s').value
    vely = obj["flash", "vely"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('km/s').value
    velz = obj["flash", "velz"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('km/s').value
    dens = obj["flash", "dens"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('g/cm**3').value
    temp = obj["flash", "temp"][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('K').value

    dz = obj['flash', 'dz'][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].to('cm').value
    mp = yt.physical_constants.mp.value  # Proton mass
    coldens = dens * dz / (1.4 * mp)

    print(f"velx.shape: {velx.shape}, vely.shape: {vely.shape}, velz.shape: {velz.shape}")
    return velx, vely, velz, coldens, temp


def visualize_velocity_field(velx, vely, velz, mask_cube, converted_points, html_root, time_Myr, vel_stride=40):
    """
    Visualize velocity field using k3d vectors.
    """
    # Masking the velocity components
    velx_masked = np.where(mask_cube, velx, np.nan)
    vely_masked = np.where(mask_cube, vely, np.nan)
    velz_masked = np.where(mask_cube, velz, np.nan)

    # Flatten and create 3D grid
    indices = np.argwhere(~np.isnan(velx_masked))
    origins = indices.astype(np.float32)

    if DEBUG:
        print(f"Original indices: {indices.shape}")

    # Randomly reduce the number of vectors to reduce clutter
    if len(indices) > 1:
        selected_indices = np.random.choice(len(indices), size=len(indices) // vel_stride, replace=False)
        indices = indices[selected_indices]
        origins = origins[selected_indices]

    # Compute raw vectors (before normalization)
    raw_vectors = np.array([
        velx_masked[indices[:, 0], indices[:, 1], indices[:, 2]],
        vely_masked[indices[:, 0], indices[:, 1], indices[:, 2]],
        velz_masked[indices[:, 0], indices[:, 1], indices[:, 2]]
    ]).T

    # Save raw vertical velocity for color mapping
    raw_velz = raw_vectors[:, 2]

    # Normalize vectors for visualization (arrow lengths/direction)
    max_vals = np.abs(raw_vectors).max(axis=0)
    vectors = raw_vectors / max_vals

    if DEBUG:
        print(f"New vectors: {vectors.shape}")
        print(f"Normalized vectors[0]: {vectors[0]}")
        print(f"min: {vectors.min()}")
        print(f"max: {vectors.max()}")

    scale = 20.0  # Adjust as needed

    # Map colors using the raw vertical velocity
    # Instead of using the full range [-1195, 1164], we use a narrower range (e.g., [-300, 300])
    # to increase the contrast: values below -300 are saturated blue, above 300 are saturated red.
    color_range = [-300, 300]  # Adjust this range based on your data distribution
    colors = k3d.helpers.map_colors(raw_velz, k3d.matplotlib_color_maps.RdBu_r, color_range)
    
    # Duplicate colors for both the vector shaft and head
    vec_colors = np.zeros(2 * len(colors))
    for i, c in enumerate(colors):
        vec_colors[2 * i] = c
        vec_colors[2 * i + 1] = c
    vec_colors = vec_colors.astype(np.uint32)

    fig = k3d.plot()
    vec = k3d.vectors(
        origins=origins - vectors / 2,
        vectors=vectors * scale,
        colors=vec_colors,
        use_head=True,
        head_size=10,
        line_width=0.1
    )
    fig += vec

    # Add explosion points
    SB_center = k3d.points(positions=np.array(converted_points, dtype=np.float32),
                             point_size=1.0,
                             shader='3d',
                             opacity=1.0,
                             color=0xc30010)
    fig += SB_center

    # Save the plot snapshot as an HTML file
    with open(os.path.join(html_root, f'{time_Myr}_vel.html'),'w') as fp:
        fp.write(fig.get_snapshot())

    if DEBUG:
        print("Done. Plot file stored at {}".format(f'{html_root}/{time_Myr}_vel.html'))


def main(args):
    for timestamp in range(args.start_timestamp, args.end_timestamp + 1, args.incr):
        time_Myr = timestamp2Myr(timestamp)
        ds = yt.load(os.path.join(args.hdf5_root, '{}{}'.format(hdf5_prefix, timestamp)))

        center = [0, 0, 0] * yt.units.pc
        arb_center = ds.arr(center, 'code_length')
        left_edge = arb_center + ds.quan(-500, 'pc')
        right_edge = arb_center + ds.quan(500, 'pc')
        obj = ds.arbitrary_grid(left_edge, right_edge, dims=(args.pixel_boundary,) * 3)

        x_range_scaled = (args.lower_bound, args.upper_bound)
        y_range_scaled = (args.lower_bound, args.upper_bound)
        z_range_scaled = (args.lower_bound, args.upper_bound)

        velx, vely, velz, dens_cube, temp_cube = get_velocity_data(obj, x_range_scaled, y_range_scaled, z_range_scaled)

        all_data = read_SNfeedback(hdf5_root=args.hdf5_root, filename=args.dat_filename)
        filtered_data = filter_data(all_data, range_coord=(-500, -500, 1000, 1000, -500, 500))
        converted_points, filtered_data = update_pos_pix256(filtered_data)

        mask_cube = np.zeros((args.pixel_boundary, args.pixel_boundary, args.upper_bound - args.lower_bound), dtype=bool)
        mask_cube = segment_cube_roi(args, dens_cube, temp_cube, mask_cube)

        visualize_velocity_field(velx, vely, velz, mask_cube, converted_points, args.html_root, time_Myr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='wholeCube_SN_target_k3d.py',
        description='Model the low density area for the entire cube, label the SN, and point out the target bubble',
        epilog='Contact Joycelyn if you dont understand')

    parser.add_argument('-hr', '--hdf5_root', help='Input the root path to where hdf5 files are stored.')
    # parser.add_argument('-m', '--mask_root', help='Input the root path to where mask files are stored.')
    parser.add_argument('-st', '--start_timestamp', help='Input the starting timestamp', type=int)
    parser.add_argument('-et', '--end_timestamp', help='Input the ending timestamp', type=int)
    parser.add_argument('-i', '--incr', help='The timestamp increment unit', default=1, type=int)
    parser.add_argument('-df', '--dat_filename', help='Input the .dat filename', default="SNfeedback.dat")
    parser.add_argument('-pixb', '--pixel_boundary', help='Input the pixel resolution', default=256, type=int)
    parser.add_argument('-lb', '--lower_bound', help='The lower bound for the cube.', default=0, type=int)
    parser.add_argument('-up', '--upper_bound', help='The upper bound for the cube.', default=256, type=int)
    parser.add_argument('-k', '--html_root', help='Input the root path to where the k3d plots should be stored')
    parser.add_argument('-Tt', '--temp_thresh', help='The power of temperature threshold for region segmentation. (hot gas)', default=5.5, type=float)
    

    args = parser.parse_args()

    main(args)
