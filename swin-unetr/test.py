import argparse
import os
from functools import partial

import nibabel as nib
import numpy as np
import torch
from utils.data_utils import get_loader, get_astro_loader

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

# python test.py --json_list=/home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/MHD-NII.json --data_dir=/home/joycelyn/Desktop/Dataset/MHD-3DIS/MHD-3DIS-NII/test --feature_size=48 --infer_overlap=0.6 --pretrained_model_name=model_final.pt --pretrained_dir=/home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs/swin-unetr-epoch300/logs --workers 0 --output_dir /home/joycelyn/Desktop/Dataset/MHD-3DIS/result-outputs/swin-unetr-epoch300/masks-output


parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--output_dir", default="/dataset/dataset0/", type=str, help="output directory")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=0, type=int, help="data fold")  # original default is 1, but somehow has to set to 0 to be reading fold 1 as testing set
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.6, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--astro_use", action="store_true", help="if training for astro purposes or not")
parser.add_argument("--timestamp", default="380", type=str, help="target timestamp")
parser.add_argument(
    "--pretrained_dir",
    default="./pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/",
    type=str,
    help="pretrained checkpoint directory",
)

def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = args.output_dir
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if args.astro_use:
        test_loader = get_astro_loader(args)
    else:
        test_loader = get_loader(args)
    
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    
    model = SwinUNETR(
        img_size=128,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    # Set up sliding window inference
    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch["image"].to(device)
            affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            
            # Get the case name from the filename
            timestamp = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split(".")[0]
            img_name = timestamp + ".nii.gz"
            
            print("Inference on case {}".format(img_name))
            # Run sliding window inference and apply sigmoid activation.
            output = model_inferer_test(image)
            activated_output = torch.sigmoid(output)
            # Aggregate the three channels by taking the maximum probability across them.
            aggregated_output = activated_output.max(dim=1, keepdim=True)[0]
            # Threshold to get a binary mask.
            binary_mask = (aggregated_output > 0.5).float()
            
            # Squeeze the batch and channel dimensions, move to CPU and convert to numpy.
            seg_array = binary_mask.squeeze().detach().cpu().numpy().astype(np.uint8)
            nib.save(nib.Nifti1Image(seg_array, affine), os.path.join(output_directory, img_name))
        print("Finished inference!")

if __name__ == "__main__":
    main()


