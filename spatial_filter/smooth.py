import os.path as osp
import math
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from filter import lin_filter, median_filter


# Smooth the image by gaussian filter.
def smooth_gaussian(img, sigma_sq, s):
    # Compute the gaussian filter.
    gaussian_kernel = np.zeros((2 * s + 1, 2 * s + 1))
    for i in range(2 * s + 1):
        for j in range(2 * s + 1):
            gaussian_kernel[i, j] = math.exp(-((i - s)**2 + (j - s)**2)/(2 * sigma_sq))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # Normalization.
    img_smoothed_gaussian = lin_filter(img.astype(float), gaussian_kernel)
    return img_smoothed_gaussian.astype(int)


# Smooth the image by median filter.
def smooth_median(img, s):
    return median_filter(img, s).astype(int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smooth an image.')
    parser.add_argument('-i', '--img_source_path', type=str, required=True, help='the path of source image.')
    parser.add_argument('-s', '--kernel_size', type=int, default=1, help='yield a kernel of size (2n+1)*(2n+1).')
    parser.add_argument('-v', '--gaussian_variance', type=float, default=1.0, help='the variance of gaussian filter.')
    args = parser.parse_args()
    img = cv2.imread(args.img_source_path, 0)
    img_smoothed_median = smooth_median(img, args.kernel_size)
    out_path_median = (osp.splitext(args.img_source_path)[0] +
                       f'_{2 * args.kernel_size + 1}x{2 * args.kernel_size + 1}_median_smoothed.jpg'
                       )
    img_smoothed_gaussian = smooth_gaussian(img, args.gaussian_variance, args.kernel_size)
    out_path_gaussian = (osp.splitext(args.img_source_path)[0] +
                         f'_{2 * args.kernel_size + 1}x{2 * args.kernel_size + 1}_{args.gaussian_variance}' +
                         '_gaussian_smoothed.jpg'
                         )

    plt.figure()

    plt.subplot(131)
    plt.title('Original')
    plt.imshow(img, cmap='gray')

    plt.subplot(132)
    plt.title('Gaussian smoothed')
    plt.imshow(img_smoothed_gaussian, cmap='gray')

    plt.subplot(133)
    plt.title('Median smoothed')
    plt.imshow(img_smoothed_median, cmap='gray')

    plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, )
    plt.show()

    cv2.imwrite(args.img_source_path, img)
    cv2.imwrite(out_path_gaussian, img_smoothed_gaussian)
    cv2.imwrite(out_path_median, img_smoothed_median)
