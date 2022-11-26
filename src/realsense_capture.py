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

frame_rate_mapping = {5, 15, 30, 60, 90}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Folder to save the captured images.", default="../images", required=False)
    parser.add_argument("--resolution", help="Image resolution (360, 480, 720)", default="480", required=False)
    parser.add_argument("--n_frames", type=int, help="Number of frames to save.", default=30, required=False)
    parser.add_argument("--frame_rate", type=int, help="Streaming frame rate (5, 15, 30, 60, 90)", default=30, required=False)
    parser.add_argument("-l", dest="stream_left_image",  help="Save left image.",  action="store_true", default=False, required=False)
    parser.add_argument("-r", dest="stream_right_image", help="Save right image.", action="store_true", default=False, required=False)
    parser.add_argument("-d", dest="stream_depth_image", help="Save depth image.", action="store_true", default=False, required=False)
    parser.add_argument("-c", dest="stream_color_image", help="Save color image.", action="store_true", default=False, required=False)
    parser.add_argument("-projector", dest="use_projector", help="Turn projector on for better depth quality", action="store_true", required=False)
    args = parser.parse_args()

    # argument checking
    if not os.path.isdir(args.output):
        raise ValueError("[FAIL]: Output path is not a directory!")
    if args.resolution not in resolution_mapping:
        raise KeyError("[FAIL]: Input resolution {} is not supported (360, 480, or 720).".format(args.resolution))
    if args.frame_rate not in frame_rate_mapping:
        raise KeyError("[FAIL]: Input frame rate {} is not supported (10, 15, or 30).".format(args.frame_rate))
    if (args.stream_left_image  or args.stream_right_image or \
        args.stream_depth_image or args.stream_color_image) == False:
        exit("[WARN]: No active stream channel selected.")

    # create a rs context object
    rs_pipeline = rs.pipeline()

    # configure streams
    (img_w, img_h) = resolution_mapping[args.resolution]
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    rs_config = rs.config()
    if args.stream_left_image:
        rs_config.enable_stream(rs.stream.infrared, 1, img_w, img_h, rs.format.y8, args.frame_rate)
    if args.stream_right_image:
        rs_config.enable_stream(rs.stream.infrared, 2, img_w, img_h, rs.format.y8, args.frame_rate)
    if args.stream_depth_image:
        rs_config.enable_stream(rs.stream.depth, img_w, img_h, rs.format.z16,  args.frame_rate)
    if args.stream_color_image:
        rs_config.enable_stream(rs.stream.color, img_w, img_h, rs.format.bgr8, args.frame_rate)

    # Start streaming
    pipeline_profile = rs_pipeline.start(rs_config)
    device = pipeline_profile.get_device()
    depth_sensor = device.query_sensors()[0]
    depth_sensor.set_option(rs.option.emitter_enabled, 1 if args.use_projector else 0)

    for i in range(args.n_frames):
        start_time = time.time()
        frames = rs_pipeline.wait_for_frames()

        if args.stream_left_image:
            frame_l = frames.get_infrared_frame(1)
            img_l = np.asanyarray(frame_l.get_data())
            cv2.imshow('RealSense', img_l)
            cv2.waitKey(1)
        if args.stream_right_image:
            frame_r = frames.get_infrared_frame(2)
            img_r = np.asanyarray(frame_r.get_data())
        if args.stream_depth_image:
            frame_d = frames.get_depth_frame()
            img_d = np.asanyarray(frame_d.get_data())
        if args.stream_color_image:
            frame_c = frames.get_color_frame()
            img_c = np.asanyarray(frame_c.get_data())

        # save images
        img_path_l = os.path.join(args.output, "{:06d}_l.png".format(i))
        img_path_r = os.path.join(args.output, "{:06d}_r.png".format(i))
        img_path_d = os.path.join(args.output, "{:06d}_d.png".format(i))
        img_path_c = os.path.join(args.output, "{:06d}_c.png".format(i))

        if args.stream_left_image:
            if cv2.imwrite(img_path_l, img_l) == False:
                exit("[FAIL]: Error encountered when saving left image '{}'".format(img_path_l))
        if args.stream_right_image:
            if cv2.imwrite(img_path_r, img_r) == False:
                exit("[FAIL]: Error encountered when saving right image '{}'".format(img_path_r))
        if args.stream_depth_image:
            if cv2.imwrite(img_path_d, img_d) == False:
                exit("[FAIL]: Error encountered when saving depth image '{}'".format(img_path_d))
        if args.stream_color_image:
            if cv2.imwrite(img_path_c, img_c) == False:
                exit("[FAIL]: Error encountered when saving color image '{}'".format(img_path_c))

        end_time = time.time()
        print("[INFO]: Frame {0} took {1} ms to process".format(i + 1, (end_time - start_time) * 1e3))


if __name__ == "__main__":
    main()