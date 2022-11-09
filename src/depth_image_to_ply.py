import os
from sys import argv

import cv2
import numpy as np

from utils import depth_to_xyz, triangulate, save_ply


def main():
    img_path = argv[1]
    img_depth = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    img_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_JET)
    h, w = img_depth.shape
    cv2.imshow("depth image ({} x {})".format(h, w), img_colormap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    K = np.array([
        [385.605,   0.000, 323.362],
        [  0.000, 385.605, 241.855],
        [  0.000,   0.000,   1.000]
    ])

    img_xyz = depth_to_xyz(K, img_depth, 1000.0)
    triangles = triangulate(img_xyz)
    save_ply(img_path.replace(".png", ".ply"), verts=img_xyz.reshape([-1, 3]), faces=triangles)


if __name__ == "__main__":
    main()