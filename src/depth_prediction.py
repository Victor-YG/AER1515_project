import os
import sys
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.abspath(os.curdir), "../PSMNet_victor"))
from models import *

from utils import load_camera_info


def find_images(folder):
    '''find all stereo images in folder'''

    images = []

    for f in os.listdir(folder):
        if "_l.png" in f:
            img_path_l = os.path.join(folder, f)
            img_path_r = img_path_l.replace("_l.png", "_r.png")
            images.append((img_path_l, img_path_r))

    return images


def load_images(img_path_l, img_path_r):
    '''load images and convert to proper form for PSMNet'''

    normal_mean_var = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**normal_mean_var)])

    img_l = Image.open(img_path_l).convert('RGB')
    img_r = Image.open(img_path_r).convert('RGB')

    img_l_n = infer_transform(img_l)
    img_r_n = infer_transform(img_r)

    img_l_p = F.pad(img_l_n, (0, 0, 0, 0)).unsqueeze(0)
    img_r_p = F.pad(img_r_n, (0, 0, 0, 0)).unsqueeze(0)

    return img_l_p, img_r_p


def disparity_to_depth(camera_matrix, img_disparity, resolution, max_disparity=192):
    '''convert disparity image to depth image'''

    h, w = img_disparity.shape
    img_depth = np.zeros_like(img_disparity, dtype=np.uint16)

    fx = camera_matrix[0, 0]
    b  = camera_matrix[2, 2]

    for v in range(h):
        for u in range(w):
            if img_disparity[v, u] > max_disparity:
                continue
            if img_disparity[v, u] == 0:
                continue
            img_depth[v, u] = np.uint16(fx * b / img_disparity[v, u] / resolution)

    return img_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", help="File containing camera info.", required=True)
    parser.add_argument("--model", help="Model parameters.", default="../PSMNet_victor/trained/pretrained_sceneflow.tar", required=False)
    parser.add_argument("--input", help="Folder containing images.", required=False)
    parser.add_argument("--img_l", help="Input left image.", required=False)
    parser.add_argument("--img_r", help="Input right image.", required=False)
    parser.add_argument("--max_disparity", help="Maximum disparity allowed.", type=int, default=192, required=False)
    parser.add_argument("--resolution", help="Resolution in mm to save the depth prediction.", type=float, default=1, required=False)
    args = parser.parse_args()

    # create model
    model = stackhourglass(args.max_disparity)
    model = nn.DataParallel(model, device_ids=[0])
    state_dict = torch.load(args.model, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['state_dict'])
    print("[INF]: Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))

    # load camera info
    cam_info = load_camera_info(args.camera)
    fx = float(cam_info["fx_l"])
    fy = float(cam_info["fy_l"])
    cx = float(cam_info["cx_l"])
    cy = float(cam_info["cy_l"])
    b  = float(cam_info["b"])
    camera_matrix = np.array([[fx,  0, cx], [ 0, fy, cy], [ 0,  0,  b]]) # hardcoded baseline at (2, 2)

    # find all images
    images = []
    if args.input is not None:
        images = find_images(args.input)

    if args.img_l is not None:
        if args.img_r is not None:
            images.append((args.img_l, args.img_r))
        else:
            images.append((args.img_l, args.img_l.replace("_l.png", "_r.png")))

    for (img_path_l, img_path_r) in images:
        # load images
        img_l, img_r = load_images(img_path_l, img_path_r)
        # print("[INFO]: input image shape: {}".format(img_l.shape))

        # run prediction
        model.eval()
        with torch.no_grad():
            img_p = model(img_l, img_r)
        img_p = torch.squeeze(img_p).data.cpu().numpy()

        # save disparity image
        arr_disparity = (img_p * 256).astype('uint16')
        img_disparity = Image.fromarray(arr_disparity)
        img_disparity.save(img_path_l.replace("_l.png", "_disparity.png"))

        # save depth image
        arr_depth = disparity_to_depth(camera_matrix, img_p, args.resolution, args.max_disparity)
        cv2.imwrite(img_path_l.replace("_l.png", "_depth.png"), arr_depth)

        print("[INFO]: Done predicting depth for '{}'".format(img_path_l))


if __name__ == "__main__":
    main()