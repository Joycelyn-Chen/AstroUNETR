import yt
import os
import argparse

parser = argparse.ArgumentParser(description="Generating Slice plots for designated timestamp range.")
parser.add_argument("-hr", "--hdf5_root", default="./Dataset", type=str, help="input hdf5 directory")
parser.add_argument("-o", "--output_root", default="./Dataset", type=str, help="output directory")
parser.add_argument('-i', '--incr', help='The timestamp increment unit', default=1, type=int)
parser.add_argument('-st', '--start_timestamp', help='Input the starting timestamp', type=int)
parser.add_argument('-et', '--end_timestamp', help='Input the ending timestamp', type=int)

def timestamp2Myr(timestamp):
    return (timestamp - 200) * 0.1 + 191

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root, exist_ok=True)

    for timestamp in range(args.start_timestamp, args.end_timestamp, args.incr):
        filename = f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}"
        time_Myr = timestamp2Myr(timestamp=timestamp)
        ds = yt.load(os.path.join(args.hdf5_root, filename))
        slc = yt.SlicePlot(ds, 'z', 'dens', center = [0, 0, 0] * yt.units.pc)
        slc.annotate_timestamp(time_unit='Myr', corner='lower_left', text_args={'size': 40,'color':'w'}) #draw_inset_box=True)
        slc.annotate_scale(corner='lower_right', text_args={'size': 40,'color':'w'}) 

        slc.save(os.path.join(args.output_root, f'{time_Myr}.png'))
