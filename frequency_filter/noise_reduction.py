import os.path as osp
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from freq_filter import dft, idft


def point_notch(img_freq, u0, v0, radius):
    # Remove a small circle at (u0, v0) from the spectrum.
    # It's equivalent to ideal highpass filter centered at (u0, v0).
    u_mat = np.array([np.arange(img_freq.shape[0]) - u0] * (img_freq.shape[1]))
    v_mat = np.array([np.arange(img_freq.shape[1]) - v0] * (img_freq.shape[0]))
    dist_sq = (u_mat ** 2).T + v_mat ** 2
    return dist_sq >= radius * radius


def notch_filter(img_freq):
    h, w = img_freq.shape
    notch = np.ones_like(img_freq)
    centers = [120, 280, 520, 680]
    for uc in centers:
        for vc in centers:
            notch *= point_notch(img_freq, h // 2 - uc, w // 2 - vc, 12)
            notch *= point_notch(img_freq, h // 2 + uc, w // 2 + vc, 12)
    return img_freq * notch


def main():
    parser = argparse.ArgumentParser(description='Smooth an image.')
    parser.add_argument('-i', '--img_source_path', type=str, default='images/Shepp-Logan.png',
                        help='the path of source image.')
    args = parser.parse_args()

    img = cv2.imread(args.img_source_path, 0)

    # DFT.
    img_ft = dft(img)
    spectrum = np.log(1 + np.abs(img_ft))
    spectrum = spectrum / np.max(spectrum) * 255

    # Noise reduction.
    img_ft_notch = notch_filter(img_ft)
    spectrum_notch = np.log(1 + np.abs(img_ft_notch))
    spectrum_notch = spectrum_notch / np.max(spectrum_notch) * 255

    # IDFT.
    img_re = idft(img_ft_notch)

    plt.figure()

    plt.subplot(221)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(222)
    plt.title('Fourier Spectrum')
    plt.imshow(spectrum, cmap='gray')
    plt.axis('off')

    plt.subplot(223)
    plt.title('after noise reduction')
    plt.imshow(img_re.astype(int), cmap='gray')
    plt.axis('off')

    plt.subplot(224)
    plt.title('Filtered Spectrum')
    plt.imshow(spectrum_notch.astype(int), cmap='gray')
    plt.axis('off')

    plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, )
    plt.show()

    cv2.imwrite(args.img_source_path, img)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_spectrum.png', spectrum)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_filter.png', spectrum_notch)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_re.png', img_re)


if __name__ == '__main__':
    main()
