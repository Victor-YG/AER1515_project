import os
import argparse

import cv2
import numpy as np
import open3d as o3d

from utils import load_camera_info, load_poses_from_aln


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", help="File path of ply file to render.", required=True)
    parser.add_argument("--camera", help="File path of camera file.", required=True)
    parser.add_argument("--aln", help="File path of aln file to indicate pose of ply file.", default=None, required=False)
    parser.add_argument("--resolution", help="Output depth resolution", type=float, default=0.1, required=False)
    parser.add_argument("--output", help="Folder to save the output depth image.", default=None, required=False)
    args = parser.parse_args()

    # check arguments
    if os.path.exists(args.ply) is False:
        exit("[FAIL]: Invalid ply file.")
    if os.path.exists(args.camera) is False:
        exit("[FAIL]: Invlid camera file.")
    if args.output is None:
        args.output = os.path.dirname(os.path.abspath(args.ply))

    # load pose of ply
    pose = np.eye(4)
    if args.aln is not None:
        pose = load_poses_from_aln(args.aln)[0]

    # load camera
    cam_info = load_camera_info(args.camera)

    # load ply file
    mesh = o3d.io.read_triangle_mesh(args.ply)

    # setup camera
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(cam_info["w"]),
        height=int(cam_info["h"]),
        fx=float(cam_info["fx_l"]),
        fy=float(cam_info["fy_l"]),
        cx=float(cam_info["cx_l"]),
        cy=float(cam_info["cy_l"]))
    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = intrinsic
    camera.extrinsic = np.linalg.inv(pose)

    # setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='depth', width=int(cam_info["w"]), height=int(cam_info["h"]))
    vis.add_geometry(mesh)

    # view controller
    controller = vis.get_view_control()
    controller.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True)

    # renderand save depth
    depth_image = vis.capture_depth_float_buffer(do_render=True)
    img_d = np.asanyarray(depth_image)
    img_d = img_d / args.resolution
    img_d = img_d.astype(np.uint16)
    cv2.imwrite(os.path.join(args.output, "saved_depth.png"), img_d)


if __name__ == "__main__":
    main()