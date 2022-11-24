import os
import argparse

import cv2
import numpy as np

from utils import load_camera_matrix, depth_to_xyz, triangulate, save_ply


def convert_depth_img_to_ply(filepath, camera_matrix):
    '''Create ply file from depth image and camera intrinsic'''

    print("[INFO]: Converting depth image '{}'...".format(filepath))
    img_depth = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)

    # img_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_JET)
    # h, w = img_depth.shape
    # cv2.imshow("depth image ({} x {})".format(h, w), img_colormap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img_xyz = depth_to_xyz(camera_matrix, img_depth, 1000.0)
    triangles = triangulate(img_xyz)
    save_ply(filepath.replace(".png", ".ply"), verts=img_xyz.reshape([-1, 3]), faces=triangles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", help="File containing camera info.", required=True)
    parser.add_argument("--input", help="Input depth image or folder.", required=True)
    args = parser.parse_args()

    camera_matrix = load_camera_matrix(args.camera)

    if os.path.isdir(args.input):
        print("[INFO]: converting depth image(s) in '{}'.".format(args.input))
        for f in os.listdir(args.input):
            if "_d.png" in f:
                convert_depth_img_to_ply(os.path.join(args.input, f), camera_matrix)
    else:
        convert_depth_img_to_ply(args.input, camera_matrix)
    print("[INFO]: Done converting depth image(s).")

if __name__ == "__main__":
    main()