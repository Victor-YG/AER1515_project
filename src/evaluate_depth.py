import os
import argparse

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="Target depth image to be evaluated.", required=True)
    parser.add_argument("--truth",  help="Ground truth depth image.", required=True)
    parser.add_argument("--resolution", help="Depth resolution of images.", type=float, default=0.1, required=False)
    parser.add_argument("--err_scale", help="Scaling for saving error heat map.", type=int, default=1000, required=False)
    args = parser.parse_args()

    # check arguments
    if os.path.exists(args.target) is False:
        exit("[FAIL]: Target depth image doesn't exist.")
    if os.path.exists(args.truth) is False:
        exit("[FAIL]: Ground truth depth image doesn't exist.")

    # read images
    img_d = cv2.imread(args.target, cv2.IMREAD_ANYDEPTH)
    img_t = cv2.imread(args.truth,  cv2.IMREAD_ANYDEPTH)
    img_d = img_d / args.resolution
    img_t = img_t / args.resolution

    # compute metrics
    img_diff = img_d - img_t
    mae  = np.sum(np.abs(img_diff))
    mse  = np.sum(np.square(img_diff))
    rmse = np.sqrt(mse)

    # save error heat map (scaled by 1000)
    img_hp = img_diff * 1000
    # TODO::convert to heatmap and save to file


if __name__ == "__main__":
    main()