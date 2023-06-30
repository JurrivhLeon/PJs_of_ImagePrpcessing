import os.path as osp
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from freq_filter import dft, idft, lowpass_gaussian, lowpass_ideal


def main(nbins=256):
    parser = argparse.ArgumentParser(description='Smooth an image.')
    parser.add_argument('-i', '--img_source_path', type=str, default='images/suomi.jpg',
                        help='the path of source image.')
    parser.add_argument('-d', '--gauss_sd', type=float, default=100,
                        help='the std. deviation of gaussian filter.')
    parser.add_argument('-r', '--radius', type=float, default=100,
                        help='the radius of ideal lowpass filter.')
    args = parser.parse_args()

    img = cv2.imread(args.img_source_path, 0)
    h, w = img.shape

    # Generate filter.
    gauss_lp = lowpass_gaussian(2 * h, 2 * w, args.gauss_sd)
    ideal_lp = lowpass_ideal(2 * h, 2 * w, args.radius)

    # DFT.
    img_ft = dft(img)

    # Original spectrum.
    spectrum = np.log(1 + np.abs(img_ft))  # Log-scale.
    spectrum = spectrum / spectrum.max() * (nbins - 1) # Normalize.

    # Smooth by gaussian filter.
    img_sm1_ft = img_ft * gauss_lp

    # New spectrum.
    spectrum_sm1 = np.log(1 + np.abs(img_sm1_ft))
    spectrum_sm1 = spectrum_sm1 / spectrum_sm1.max() * (nbins - 1)

    # IDFT.
    img_sm1 = idft(img_sm1_ft)

    # Smooth by ideal filter.
    img_sm2_ft = img_ft * ideal_lp

    # New spectrum.
    spectrum_sm2 = np.log(1 + np.abs(img_sm2_ft))
    spectrum_sm2 = spectrum_sm2 / spectrum_sm2.max() * (nbins - 1)

    # IDFT.
    img_sm2 = idft(img_sm2_ft)

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
    plt.title('Smoothed by Gaussian')
    plt.imshow(img_sm1, cmap='gray')
    plt.axis('off')

    plt.subplot(235)
    plt.title('Smoothed Spectrum (log-scale)')
    plt.imshow(spectrum_sm1, cmap='gray')
    plt.axis('off')

    plt.subplot(233)
    plt.title('Smoothed by Ideal')
    plt.imshow(img_sm2, cmap='gray')
    plt.axis('off')

    plt.subplot(236)
    plt.title('Smoothed Spectrum (log-scale)')
    plt.imshow(spectrum_sm2, cmap='gray')
    plt.axis('off')

    plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, )
    plt.show()

    cv2.imwrite(args.img_source_path, img)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_spectrum.jpg', spectrum)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_gauss_lp_spectrum.jpg', spectrum_sm1)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_ideal_lp_spectrum.jpg', spectrum_sm2)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_gaussian_smoothed.jpg', img_sm1)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_ideal_smoothed.jpg', img_sm2)


if __name__ == '__main__':
    main()
