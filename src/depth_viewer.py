from email import parser
import cv2
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help="Input depth image to be visualized.", required=True)
    parser.add_argument("--o", help="Output color image to be saved.", default=None, required=False)
    parser.add_argument("--s", help="Scaling to convert depth to color.", type=float, default=0.05, required=False)
    args = parser.parse_args()

    img_i = cv2.imread(args.i, cv2.IMREAD_ANYDEPTH)
    img_o = cv2.applyColorMap(cv2.convertScaleAbs(img_i, alpha=args.s), cv2.COLORMAP_VIRIDIS)
    cv2.imshow("colormap", img_o)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.o is not None:
        print("[INFO]: Saving image as '{}'".format(args.o))
        cv2.imwrite(args.o, img_o)


if __name__ == "__main__":
    main()