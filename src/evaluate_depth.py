import os
import argparse

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="Target depth image to be evaluated.", required=True)
    parser.add_argument("--image", help="Target image to render heatmap on top.", default=None, required=False)
    parser.add_argument("--truth",  help="Ground truth depth image.", required=True)
    parser.add_argument("--resolution", help="Depth resolution of images.", type=float, default=0.1, required=False)
    parser.add_argument("--err_scale", help="Scaling for saving error heat map.", type=int, default=1000, required=False)
    parser.add_argument("--output", help="Folder to output the heatmap.", required=True)
    args = parser.parse_args()

    # check arguments
    if os.path.exists(args.target) is False:
        exit("[FAIL]: Target depth image doesn't exist.")
    if os.path.exists(args.truth) is False:
        exit("[FAIL]: Ground truth depth image doesn't exist.")

    # read images
    img_d = cv2.imread(args.target, cv2.IMREAD_ANYDEPTH)
    img_t = cv2.imread(args.truth,  cv2.IMREAD_ANYDEPTH)
    img_d = img_d.astype(np.float64)
    img_t = img_t.astype(np.float64)
    img_d = img_d * args.resolution
    img_t = img_t * args.resolution

    # compute mask
    mask_1 = (img_d == 0)
    mask_2 = (img_t == 0)
    mask = np.ma.mask_or(mask_1, mask_2)

    # compute metrics
    img_diff = img_t - img_d
    img_diff[mask] = 0
    count = img_diff[mask].shape
    mae  = np.sum(np.abs(img_diff)) / count
    mse  = np.sum(np.square(img_diff)) / count
    rmse = np.sqrt(mse)
    print("[INFO]: mae = {}, mse = {}, rmse = {}".format(mae, mse, rmse))

    # save error heat map (scaled)
    img_diff = img_diff * args.err_scale
    img_heatmap = cv2.applyColorMap(cv2.convertScaleAbs(img_diff, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.output, "err_heatmap_s={}.png".format(args.err_scale)), img_heatmap)


if __name__ == "__main__":
    main()