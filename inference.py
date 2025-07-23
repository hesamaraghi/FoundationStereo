from pathlib import Path
from glob import glob
import os
import numpy as np
from rectify_event_depth import Pixel_Projector

import sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *
import torch
from tqdm import tqdm

test_path = Path(__file__).parent / "data" / "NSEK" / "test"

skip_number = 5

ckpt_dir = "pretrained_models/23-51-11/model_best_bp2.pth"
saving_dir = "output"
scale = 1
hiera = 0
valid_iters = 32

camera_type = {
    "K_SXW": "435",
    "K_EWI": "455",
    "K_Alejandro": "455",
    "K_SGHC": "435",
}

set_logging_format()
torch.autograd.set_grad_enabled(False)
out_dir = Path(saving_dir) / "visualize" / "pred" / "test"
os.makedirs(out_dir, exist_ok=True)

cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
args = OmegaConf.create(cfg)
logging.info(f"args:\n{args}")
logging.info(f"Using pretrained model from {ckpt_dir}")

model = FoundationStereo(args)

ckpt = torch.load(ckpt_dir)
logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
model.load_state_dict(ckpt['model'])

model.cuda()
model.eval()

activity_list = [
    a for a in test_path.iterdir() if a.is_dir() and not a.name.startswith(".")
]

for activity in activity_list:
    print(f"Processing activity: {activity.name}")
    splits = activity.name.split("_")
    kitchen = splits[0] + "_" + splits[1]
    activity_name = "_".join(splits[2:])
    print(f"Kitchen: {kitchen}, Activity Name: {activity_name}")
    activity_path = test_path / activity
    csv_file = activity_path / (activity.name + ".csv")
    with open(csv_file, "r") as file:
        lines = file.readlines()
    lines = [l.strip() for l in lines[1::skip_number]]
    activity_output_dir = out_dir / activity.name
    os.makedirs(activity_output_dir, exist_ok=True)

    assert kitchen in camera_type, f"Kitchen {kitchen} not found in camera_type mapping"
    calibration_path = (
        Path(__file__).parent
        / "data"
        / "Calibration"
        / "calibration_results"
        / ("calibration_with_" + camera_type[kitchen])
    )
    reconstructed_data_path = (
        Path(__file__).parent
        / "data"
        / "Reconstruction"
        / kitchen
        / activity_name
    )
    assert (
        calibration_path.exists()
    ), f"Calibration path {calibration_path} does not exist"
    assert (
        reconstructed_data_path.exists()
    ), f"Reconstructed data path {reconstructed_data_path} does not exist"
    
    projector = Pixel_Projector(
        calibration_path=calibration_path,
        data_path=reconstructed_data_path,
    )
    
    baseline = np.linalg.norm(projector.translation_matrix_EVENT0_EVENT1)
    print(f"Baseline: {baseline} milimeters")
    
    print(f"Number of reconstructed EVENT0 samples: {len(projector.EVENT0_ts)}")
    print(f"First Sample EVENT0_ts: {projector.EVENT0_ts[0]}, {projector.EVENT0_data_path_list[0]}")
    print(f"Last Sample EVENT0_ts:  {projector.EVENT0_ts[-1]}, {projector.EVENT0_data_path_list[-1]}")
    
    print(f"Number of reconstructed EVENT1 samples: {len(projector.EVENT1_ts)}")
    print(f"First Sample EVENT1_ts: {projector.EVENT1_ts[0]}", projector.EVENT1_data_path_list[0])
    print(f"Last Sample EVENT1_ts:  {projector.EVENT1_ts[-1]}", projector.EVENT1_data_path_list[-1])

    for line in tqdm(lines):
        parts = line.split(",")
        assert len(parts) == 2, "Expected two parts in the line"
        timestamp_str = parts[0].strip()
        timestamp_sec_str = timestamp_str[:-6] + "." + timestamp_str[-6:]
        timestamp = np.asarray(timestamp_sec_str, dtype=np.float64)
        idx = parts[1].strip()
        # print(f"Processing timestamp: {timestamp}, idx: {idx}", flush=True, end=", ")
        disp_gt_path = activity_path / "disparity" / "event" / (idx.zfill(6) + ".png")
        assert disp_gt_path.exists(), f"Disparity ground truth path {disp_gt_path} does not exist"
        left_rectified, right_rectified = projector.rectify(timestamp)
        # print(f"Left Rectified Shape: {left_rectified.shape}, Right Rectified Shape: {right_rectified.shape}")
        H,W = left_rectified.shape[:2]
        
        left_rectified = left_rectified.copy()
        left_rectified = torch.as_tensor(left_rectified).cuda().float()[None].permute(0,3,1,2)
        right_rectified = torch.as_tensor(right_rectified).cuda().float()[None].permute(0,3,1,2)
        padder = InputPadder(left_rectified.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(left_rectified, right_rectified)

        with torch.amp.autocast('cuda', enabled=True):
            disp = model.forward(img0, img1, iters=valid_iters, test_mode=True)
        
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H,W)
        assert not np.any(disp == 0), "Disparity map contains zero values"
        depth = baseline * projector.EVENT0_intrinsic_matrix[0,0]  / disp 
        depth[depth < 200] = 200
        depth[depth > 1500] = 1500
        imageio.imwrite(activity_output_dir / (idx.zfill(6) + ".png"), depth.astype(np.uint16))
        # logging.info(f"Output saved to {out_dir}")
        








