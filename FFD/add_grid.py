import os.path as osp
import numpy as np
import argparse
import cv2
from ffd import FFD


def add_grid(tfm: FFD, img):
    img_grid = np.zeros_like(img)
    img_grid[:, :, :] = img
    for x in np.linspace(0, tfm.lx * (tfm.ncpx - 1), tfm.ncpx).astype(int):
        img_grid[x, :, :] = 0
    for y in np.linspace(0, tfm.ly * (tfm.ncpy - 1), tfm.ncpy).astype(int):
        img_grid[:, y, :] = 0
    return img_grid


def main():
    parser = argparse.ArgumentParser(description='Add grid.')
    parser.add_argument('-i', '--img_path', type=str, default='images/cat.png',
                        help='the path of source image.')
    args = parser.parse_args()

    img = cv2.imread(args.img_path)
    if img.shape[0] != 513 or img.shape[1] != 513:
        img = cv2.resize(img, (513, 513), interpolation=cv2.INTER_AREA)
    tfm = FFD()
    img_grid = add_grid(tfm, img)
    cv2.imwrite(osp.splitext(args.img_path)[0] + '_grid.png', img_grid)


if __name__ == '__main__':
    main()
