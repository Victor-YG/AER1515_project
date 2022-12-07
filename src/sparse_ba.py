import os
import argparse

import numpy as np
from utils import load_camera_info, load_poses, load_points, load_constraints
from g2o_wrapper import StereoBundleAdjustment


'''
main function for sparse ba and pose graph optimization.
Here we use a trick to avoid loop detection.
Assuming the first and last frame are close enough so that it can find enough correspondences.
Then, We manually duplicated the first frame as the last frame.
Then we can force the new last frame to be the same with the first frame with very high weight.
We allow user to specify whether there is a loop or not.
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", help="File containing camera info.", required=True)
    parser.add_argument("--poses",  help="File path of poses.txt.", required=True)
    parser.add_argument("--points", help="File path of points.txt", required=True)
    parser.add_argument("--constraints", help="File path of constraints.txt", required=True)
    parser.add_argument("--window", help="Size of sliding window", type=int, default=2, required=False)
    parser.add_argument("--loop", help="To indicate whether the last frame overlap the first frame.", default=False, required=False)
    parser.add_argument("--output", help="Output folder.", default=None, required=False)
    args = parser.parse_args()

    # read inputs
    poses = load_poses(args.poses)
    points = load_points(args.points)
    constraints = load_constraints(args.constraints)

    # load camera info
    cam_info = load_camera_info(args.camera)
    fx = float(cam_info["fx_l"])
    fy = float(cam_info["fy_l"])
    cx = float(cam_info["cx_l"])
    cy = float(cam_info["cy_l"])
    b  = float(cam_info["b"])
    camera_matrix = np.array([[fx,  0, cx], [ 0, fy, cy], [ 0,  0,  b]]) # hardcoded baseline at (2, 2)

    # sliding window local BA
    swba = StereoBundleAdjustment()
    swba.define_optimization(camera_matrix, poses, points, constraints)

    swba.optimize(max_iterations=50)
    poses_optimized  = swba.get_poses()
    points_optimized = swba.get_points()

    # pose graph optimization


    # global BA

    print("[INFO]: Done optimization.")

if __name__ == "__main__":
    main()