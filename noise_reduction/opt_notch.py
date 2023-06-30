import os.path as osp
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt


def point_notch(h, w, u0, v0, sd=5):
    # Remove a small circle at (u0, v0) from the spectrum.
    # It's equivalent to Gaussian highpass filter centered at (u0, v0).
    u_mat = np.array([np.arange(h) - u0] * w)
    v_mat = np.array([np.arange(w) - v0] * h)
    dist_sq = (u_mat ** 2).T + v_mat ** 2
    return 1 - np.exp(-dist_sq / (2 * sd * sd))


def notch_pass(h, w, u0, v0, sd=5):
    # Yield a notch pass filter.
    nr = np.ones((h, w))
    if len(u0) - len(v0):
        raise ValueError
    for i in range(len(u0)):
        nr *= point_notch(h, w, u0[i], v0[i], sd)
        nr *= point_notch(h, w, h - 1 - u0[i], w - 1 - v0[i], sd)
    cv2.imwrite('images/notch.jpg', (1 - nr) * 255)
    return 1 - nr


def noise_mod(img, notch):
    # Simulate the noise pattern in space domain.
    h, w = img.shape

    # Observe the Fourier spectrum of the image.
    img_pad = np.zeros((2 * h, 2 * w))
    sign = -np.ones((2 * h, 2 * w))
    for i in range(2 * h):
        sign[i, i % 2::2] = 1
    img_pad[:h, :w] = img
    img_ft = np.fft.fft2(img_pad * sign)

    spectrum = np.log(1 + np.abs(img_ft))
    spectrum = spectrum / np.max(spectrum) * 255

    plt.figure()
    # plt.scatter(x=[356, 588, 628, 784, 842, 1056, 1114, 1328],
    #            y=[384, 104, 492, 140, 370, 248, 478, 356], marker='x')
    plt.imshow(spectrum.astype(int), cmap='gray')
    plt.show()
    cv2.imwrite('images/spectrum.jpg', spectrum)

    # Generate the noise pattern.
    noise_ft = img_ft * notch
    noise_pad = np.fft.ifft2(noise_ft) * sign
    return np.real(noise_pad[:h, :w])


def moving_avg(mat, size=1):
    # Compute the average of a neighborhood while moving on a matrix.
    avg = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            a, b = max(i - size, 0), min(i + size + 1, mat.shape[0])
            l, r = max(j - size, 0), min(j + size + 1, mat.shape[1])
            avg[i, j] = np.mean(mat[a:b, l:r])
    return avg


def opt_filter(img, noise, size=1, nbins=256):
    # Compute the weight matrix.
    avg_noise = moving_avg(noise, size)
    sxy = moving_avg(img * noise, size) - moving_avg(img, size) * avg_noise
    sxx = moving_avg(noise * noise, size) - avg_noise ** 2
    sxx[sxx == 0] = 1e-5  # Avoid dividing by 0.

    # Denoise.
    img_denoise = img - noise * sxy / sxx
    # Cut-off.
    img_denoise[img_denoise >= nbins] = nbins - 1
    img_denoise[img_denoise < 0] = 0
    return img_denoise.astype(int)


def main():
    parser = argparse.ArgumentParser(description='Add white noises to an image.')
    parser.add_argument('-i', '--img_source_path', type=str, default='images/heart_interf.jpg',
                        help='the path of source image.')
    parser.add_argument('-s', '--size', type=int, default=1,
                        help='the size of neighborhood in optimal notch filter.')
    args = parser.parse_args()

    img = cv2.imread(args.img_source_path, 0)
    # Centers of notch filters.
    # Reference arguments:
    # u0 = [384, 104, 492, 140, 370, 248, 478, 356]
    # v0 = [356, 588, 628, 784, 842, 1056, 1114, 1328] (images/heart.jpg)
    # u0 = [692, 680]
    # v0 = [308, 874] (images/zelenskyy.jpg)
    # You need to set u0, v0 in line 99-100 manually.
    u0 = []
    v0 = []
    notch = notch_pass(2 * img.shape[0], 2 * img.shape[1], u0, v0)
    noise = noise_mod(img, notch)
    noise_pattern = (noise - noise.min()) / (noise.max() - noise.min()) * 255
    img_dn = opt_filter(img, noise, size=args.size)

    plt.figure()

    plt.subplot(221)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(222)
    plt.title('Noise Pattern')
    plt.imshow(noise_pattern, cmap='gray')
    plt.axis('off')

    plt.subplot(223)
    plt.title('Denoised')
    plt.imshow(img - noise, cmap='gray')
    plt.axis('off')

    plt.subplot(224)
    plt.title('Denoised (Opt.)')
    plt.imshow(img_dn, cmap='gray')
    plt.axis('off')

    plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, )
    plt.show()

    cv2.imwrite(args.img_source_path, img)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_noisepattern.jpg', noise_pattern)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_denoised.jpg', img - noise)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_opt_denoised.jpg', img_dn)


if __name__ == '__main__':
    main()
