import os
import time
import argparse
import multiprocessing

import cv2
import numpy as np
from PIL import Image

import pyrealsense2 as rs


resolution_mapping = {
    "360" : ( 480, 360),
    "480" : ( 640, 480),
    "720" : (1280, 720)
}

frame_rate_mapping = {10, 15, 30}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Folder to save the captured images.", default="../images", required=False)
    parser.add_argument("--resolution", help="Image resolution (360, 480, 720)", default="480", required=False)
    parser.add_argument("--n_frames", type=int, help="Number of frames to save.", default=30, required=False)
    parser.add_argument("--frame_rate", type=int, help="Streaming frame rate (10, 15, 30)", default=30, required=False)
    # parser.add_argument("--settings", help="JSON file for camera settings.", required=True)
    parser.add_argument("-r", dest="stream_right_image", help="Save right image.", action="store_true", default=False, required=False)
    parser.add_argument("-d", dest="stream_depth_image", help="Save depth image.", action="store_true", default=False, required=False)
    parser.add_argument("-c", dest="stream_color_image", help="Save color image.", action="store_true", default=False, required=False)
    args = parser.parse_args()

    # argument checking
    if not os.path.isdir(args.output):
        raise ValueError("[FAIL]: Output path is not a directory!")
    if args.resolution not in resolution_mapping:
        raise KeyError("[FAIL]: Input resolution {} is not supported (360, 480, or 720).".format(args.resolution))
    if args.frame_rate not in frame_rate_mapping:
        raise KeyError("[FAIL]: Input frame rate {} is not supported (10, 15, or 30).".format(args.frame_rate))

    # create a rs context object
    rs_pipeline = rs.pipeline()

    # configure streams
    (img_w, img_h) = resolution_mapping[args.resolution]
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.infrared, 1, img_w, img_h, rs.format.y8, args.frame_rate)

    if args.stream_right_image:
        rs_config.enable_stream(rs.stream.infrared, 2, img_w, img_h, rs.format.y8, args.frame_rate)
    if args.stream_depth_image:
        rs_config.enable_stream(rs.stream.depth, img_w, img_h, rs.format.z16,  args.frame_rate)
    if args.stream_color_image:
        rs_config.enable_stream(rs.stream.color, img_w, img_h, rs.format.bgr8, args.frame_rate)

    # Start streaming
    rs_pipeline.start(rs_config)

    for i in range(args.n_frames):
        start_time = time.time()
        frames = rs_pipeline.wait_for_frames()

        frame_l = frames.get_infrared_frame(1)
        arr_l = np.asanyarray(frame_l.get_data())
        cv2.imshow('RealSense', arr_l)
        cv2.waitKey(1)

        if args.stream_right_image:
            frame_r = frames.get_infrared_frame(2)
            arr_r = np.asanyarray(frame_r.get_data())
        if args.stream_depth_image:
            frame_d = frames.get_depth_frame()
            arr_d = np.asanyarray(frame_d.get_data())
        if args.stream_color_image:
            frame_c = frames.get_color_frame()
            arr_c = np.asanyarray(frame_c.get_data())

        # save images
        img_path_l = os.path.join(args.output, "{:06d}_l.png".format(i))
        img_path_r = os.path.join(args.output, "{:06d}_r.png".format(i))
        img_path_d = os.path.join(args.output, "{:06d}_d.png".format(i))
        img_path_c = os.path.join(args.output, "{:06d}_c.png".format(i))

        if cv2.imwrite(img_path_l, arr_l) == False:
            exit("[FAIL]: Error encountered when saving left image '{}'".format(img_path_l))

        if args.stream_right_image:
            if cv2.imwrite(img_path_r, arr_r) == False:
                exit("[FAIL]: Error encountered when saving right image '{}'".format(img_path_r))
        if args.stream_depth_image:
            if cv2.imwrite(img_path_d, arr_d) == False:
                exit("[FAIL]: Error encountered when saving depth image '{}'".format(img_path_d))
        if args.stream_color_image:
            if cv2.imwrite(img_path_c, arr_c) == False:
                exit("[FAIL]: Error encountered when saving color image '{}'".format(img_path_c))

        end_time = time.time()
        print("[INFO]: Frame {0} took {1} ms to process".format(i + 1, (end_time - start_time) * 1e3))


if __name__ == "__main__":
    main()