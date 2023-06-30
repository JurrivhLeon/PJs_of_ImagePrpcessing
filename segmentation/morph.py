import os.path as osp
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt


def circle(r):
    # Construct a circle with radius r.
    circ = np.zeros((2 * r + 1, 2 * r + 1))
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            if (i - r) ** 2 + (j - r) ** 2 <= r ** 2:
                circ[i, j] = 1
    return circ.astype(bool)


def pad(img, r):
    # Zero-padding.
    h, w = img.shape
    img_pad = np.zeros((h + 2 * r, w + 2 * r))
    img_pad[r: h + r, r: w + r] = img
    return img_pad


def dilation(img, r):
    # Dilation the object in the image.
    circ = circle(r)
    img_pad = pad(img, r)
    dil = np.zeros_like(img)

    # Traverse all pixels.
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Detect intersection.
            block = img_pad[i: i + 2 * r + 1, j: j + 2 * r + 1]
            dil[i, j] = np.any(block[circ])
    return dil.astype(int)


def erosion(img, r):
    # Erosion the object in the image.
    circ = circle(r)
    img_pad = pad(img, r)
    ero = np.zeros_like(img)

    # Traverse all pixels.
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Detect inclusion.
            block = img_pad[i: i + 2 * r + 1, j: j + 2 * r + 1]
            ero[i, j] = np.all(block[circ])
    return ero.astype(int)


def imgopen(img, r):
    ero = erosion(img, r)
    dil = dilation(ero, r)
    return dil


def imgclose(img, r):
    dil = dilation(img, r)
    ero = erosion(dil, r)
    return ero


def main():
    img_path = 'images/zmic_fdu_noise.jpg',

    img = cv2.imread(img_path, 0)
    # Binarize.
    img_thr = (1 - img / 255).astype(int)

    # Repair step by step.
    # img_re = dilation(img_thr, r=3)
    # cv2.imwrite(osp.splitext(img_path)[0] + f'_1.jpg', 255 * (1 - img_re))
    # img_re = erosion(img_re, r=3)
    # cv2.imwrite(osp.splitext(img_path)[0] + f'_2.jpg', 255 * (1 - img_re))
    # img_re = erosion(img_re, r=2)
    # cv2.imwrite(osp.splitext(img_path)[0] + f'_3.jpg', 255 * (1 - img_re))
    # img_re = dilation(img_re, r=2)
    # cv2.imwrite(osp.splitext(img_path)[0] + f'_4.jpg', 255 * (1 - img_re))

    # Restore.
    img_re = imgclose(img_thr, r=3)
    img_re = imgopen(img_re, r=2)
    # De-binarize.
    img_re = 255 * (1 - img_re)

    # Visualization.
    plt.figure()

    plt.subplot(211)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(212)
    plt.title('Repaired')
    plt.imshow(img_re, cmap='gray')
    plt.axis('off')

    plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, )
    plt.show()

    cv2.imwrite(osp.splitext(img_path)[0] + f'_repair.jpg', img_re)


if __name__ == '__main__':
    main()