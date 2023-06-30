import os.path as osp
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from freq_filter import dft, idft, laplacian, highpass_gaussian


def main(nbins=256):
    parser = argparse.ArgumentParser(description='Smooth an image.')
    parser.add_argument('-i', '--img_source_path', type=str, default='images/moon.jpg',
                        help='the path of source image.')
    parser.add_argument('-c', '--weight', type=float, default=1,
                        help='the weight of laplacian and gaussian filter.')
    parser.add_argument('-d', '--gauss_sd', type=float, default=100,
                        help='the std. deviation of gaussian filter.')
    args = parser.parse_args()

    img = cv2.imread(args.img_source_path, 0)
    h, w = img.shape

    # Generate filters.
    laplace_f = laplacian(2 * h, 2 * w)
    gauss_hp = highpass_gaussian(2 * h, 2 * w, args.gauss_sd)

    # DFT.
    img_ft = dft(img)

    # Original spectrum.
    spectrum = np.log(1 + np.abs(img_ft))  # Log-scale.
    spectrum = spectrum / spectrum.max() * (nbins - 1)  # Normalize.

    # Sharpen by Laplacian filter.
    img_sh1_ft = img_ft * (1 - args.weight * laplace_f)
    spectrum_sh1 = np.log(1 + np.abs(img_sh1_ft))
    spectrum_sh1 = spectrum_sh1 / spectrum_sh1.max() * (nbins - 1)

    # IDFT and cut off the pixels out of range [0,255].
    img_sh1 = idft(img_sh1_ft)
    img_sh1[img_sh1 >= nbins] = nbins - 1
    img_sh1[img_sh1 < 0] = 0

    # High-boost by Gaussian filter.
    img_sh2_ft = img_ft * (1 + args.weight * gauss_hp)
    spectrum_sh2 = np.log(1 + np.abs(img_sh2_ft))
    spectrum_sh2 = spectrum_sh2 / spectrum_sh2.max() * (nbins - 1)

    # IDFT and cut off the pixels out of range [0,255].
    img_sh2 = idft(img_sh2_ft)
    img_sh2[img_sh2 >= nbins] = nbins - 1
    img_sh2[img_sh2 < 0] = 0

    plt.figure()

    plt.subplot(231)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(234)
    plt.title('Fourier Spectrum (log-scale)')
    plt.imshow(spectrum, cmap='gray')
    plt.axis('off')

    plt.subplot(232)
    plt.title('Sharpened by Laplacian')
    plt.imshow(img_sh1, cmap='gray')
    plt.axis('off')

    plt.subplot(235)
    plt.title('Sharpened Spectrum (log-scale)')
    plt.imshow(spectrum_sh1, cmap='gray')
    plt.axis('off')

    plt.subplot(233)
    plt.title('Sharpened by Gaussian')
    plt.imshow(img_sh2, cmap='gray')
    plt.axis('off')

    plt.subplot(236)
    plt.title('Sharpened Spectrum (log-scale)')
    plt.imshow(spectrum_sh2, cmap='gray')
    plt.axis('off')

    plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, )
    plt.show()

    cv2.imwrite(args.img_source_path, img)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_spectrum.jpg', spectrum)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_lap_spectrum.jpg', spectrum_sh1)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_hb_spectrum.jpg', spectrum_sh2)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_laplacian_sharpened.jpg', img_sh1)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_highboost_sharpened.jpg', img_sh2)


if __name__ == '__main__':
    main()
