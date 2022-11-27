import os
import sys
import time
import argparse

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.abspath(os.curdir), "../tsdf-fusion-python_victor"))
import fusion
from utils import load_poses, load_camera_matrix


def load_images(folder, resolution, max_depth):
    '''load the grayscale/rgb images and depth images'''

    imgs_color = []
    imgs_depth = []

    files = os.listdir(folder)
    files.sort()

    for f in files:
        if "_l.png" in f:
            filepath = os.path.join(folder, f)
            img_color = cv2.imread(filepath, cv2.COLOR_GRAY2RGB)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2RGB)

            img_path_d = filepath.replace("_l.png", "_d.png")
            if os.path.exists(img_path_d) == False:
                img_path_d = filepath.replace("_l.png", "_depth.png")

            img_depth = cv2.imread(img_path_d, cv2.IMREAD_ANYDEPTH)
            img_depth = img_depth.astype(np.float64)
            img_depth *= resolution

            msk = img_depth > max_depth
            img_depth[msk] = 0.0
            imgs_color.append(img_color)
            imgs_depth.append(img_depth)


    return imgs_color, imgs_depth


def construct_volume(camera_matrix, imgs_depth, poses):
    '''find the bound of volume for reconstruction'''

    bounds = np.zeros([3, 2])

    # Compute camera view frustum and extend convex hull
    for (img_depth, pose) in zip(imgs_depth, poses):
        view_frust_pts = fusion.get_view_frustum(img_depth, camera_matrix, pose)
        bounds[:, 0] = np.minimum(bounds[:, 0], np.amin(view_frust_pts, axis=1))
        bounds[:, 1] = np.maximum(bounds[:, 1], np.amax(view_frust_pts, axis=1))

    return bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Folder to save output.", required=True)
    parser.add_argument("--images", help="Folder containing captured images.", required=True)
    parser.add_argument("--camera", help="File containing camera information", required=True)
    parser.add_argument("--poses",  help="Poses.txt containing all of the poses.", required=True)
    parser.add_argument("--decay",  help="Rate of decay is a cummulative multiple.", default=0.95, required=False)
    parser.add_argument("--resolution", help="mm of each depth increment.", type=float, default=1, required=False)
    parser.add_argument("--max_d",  help="Max value allowed for depth.", type=float, default=500, required=False)
    parser.add_argument("--voxel_size", help="Voxel size in mm.", default=3, required=False)
    #TODO::add parameter for voxel size
    args = parser.parse_args()

    # load input
    imgs_color, imgs_depth = load_images(args.images, args.resolution, args.max_d)
    poses = load_poses(args.poses)
    camera_matrix = load_camera_matrix(args.camera)

    # construct volume
    volume_bounds = construct_volume(camera_matrix, imgs_depth, poses)

    # depth fusion
    tsdf_vol = fusion.TSDFVolume(volume_bounds, voxel_size=args.voxel_size)
    weight = 1.0

    i = 0
    for (img_depth, img_color, pose) in zip(imgs_depth, imgs_color, poses):
        weight *= args.decay
        tsdf_vol.integrate(img_color, img_depth, camera_matrix, pose, obs_weight=weight)
        print("[INFO]: integrating ...")
        # i += 1
        # if i == 2:
        #     break

    # save mesh
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite(os.path.join(args.output, "reconstruction.ply"), verts, faces, norms, colors)


if __name__ == "__main__":
    main()