import os.path as osp
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt


def period_noise(h, w, amp=2.5e+04, nbins=256):
    # Generate random periodical noise term.
    noise_ft = np.zeros((h, w))
    t = 2 + np.abs(np.random.randn(4))
    u1, v1 = int(h / t[0]), int(w / t[1])
    u2, v2 = int(h / t[2]), int(w / t[3])
    # Amplify two pairs of symmetric points in frequency domain.
    noise_ft[u1, v1] = amp
    noise_ft[h - u1, w - v1] = amp
    noise_ft[u2, w - v2] = amp
    noise_ft[h - u2, v2] = amp
    # Centralize.
    sign = -np.ones((h, w))
    for i in range(h):
        sign[i, i % 2::2] = 1
    noise = np.fft.ifft2(noise_ft) * sign * (nbins - 1)
    return np.real(noise).astype(int)


def pollute_prd(img, amp=2.5e+04, nbins=256):
    # Pollute an image with periodic noise.
    h, w = img.shape
    img_prd = img + period_noise(h, w, amp=amp)
    # Cut-off.
    img_prd[img_prd >= nbins] = nbins - 1
    img_prd[img_prd < 0] = 0
    return img_prd.astype(int)


def white_noise(h, w, amp=2.5e+04):
    # Generate random white noise term.
    # Randomly generate a base on Uniform(0,1).
    base = np.random.rand(h, w)
    base_ft = np.fft.fft2(base)
    # Uniformize the spectrum to get a phase matrix.
    phase = base_ft / np.abs(base_ft)
    noise = amp * np.fft.ifft2(phase)
    return np.real(noise).astype(int)


def pollute_wn(img, amp=2.5e+04, nbins=256):
    # Pollute an image with white noise.
    h, w = img.shape
    img_wn = img + white_noise(h, w, amp=amp)
    # Cut-off.
    img_wn[img_wn >= nbins] = nbins - 1
    img_wn[img_wn < 0] = 0
    return img_wn.astype(int)


def pollute_gn(img, mu=0, amp=25, nbins=256):
    # Pollute an image with Gaussian noise.
    h, w = img.shape
    img_gn = img + (np.random.randn(h, w) + mu) * amp
    img_gn[img_gn >= nbins] = nbins - 1
    img_gn[img_gn < 0] = 0
    return img_gn.astype(int)


def pollute_pn(img, pepper=0.03, salt=0.01, nbins=256):
    # Pollute an image with pepper-salt noise.
    h, w = img.shape
    img_pn = np.zeros_like(img)
    peppersalt = np.random.rand(h, w)
    img_pn[:, :] = img
    img_pn[peppersalt <= pepper] = 0
    img_pn[peppersalt > 1 - salt] = nbins - 1
    return img_pn.astype(int)


def main():
    parser = argparse.ArgumentParser(description='Add white noises to an image.')
    parser.add_argument('-i', '--img_source_path', type=str, default='images/heart.jpg',
                        help='the path of source image.')
    args = parser.parse_args()

    img = cv2.imread(args.img_source_path, 0)
    img_wn = pollute_wn(img)
    img_prd = pollute_prd(img)
    img_gn = pollute_gn(img)
    img_pn = pollute_pn(img)

    img_dict = {'White Noise': img_wn,
                'Periodic Noise': img_prd,
                'Gaussian Noise': img_gn,
                'Pepper-Salt Noise': img_pn
                }

    for k, v in img_dict.items():
        plt.figure()

        plt.subplot(121)
        plt.title('Original')
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(122)
        plt.title('Added ' + k)
        plt.imshow(v, cmap='gray')
        plt.axis('off')

        plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, )
        plt.show()

    cv2.imwrite(args.img_source_path, img)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_wn.jpg', img_wn)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_prd.jpg', img_prd)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_gn.jpg', img_gn)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_pn.jpg', img_pn)


if __name__ == '__main__':
    main()
