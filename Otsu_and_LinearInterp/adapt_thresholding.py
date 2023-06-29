import numpy as np
import os.path as osp
import cv2
import argparse
import matplotlib.pyplot as plt
from local_hist import local_histogram

def glb_otsu(img, nbins=256):
    hist = np.array([0]*nbins)
    for k in range(nbins):
        hist[k] = (img == k).sum()
    cdf = np.cumsum(hist)
    npixels = cdf[-1]
    cdf = cdf / npixels  # Compute the cdf from 0 to L-1, and normalize it to interval [0,1].
    acm = np.cumsum(np.array(range(nbins)) * hist / npixels)  # Compute the accumulated average from 0 to L-1.
    mean_glb = acm[-1]  # The global mean.
    w1w2 = cdf * (1 - cdf)
    w1w2[w1w2 <= 1e-6] = 1  # Preprocess data to avoid the expression below being divided by 0.
    dsq_bet = (mean_glb * cdf - acm) ** 2 / w1w2  # Compute the between-classes variance at threshold from 0 to L-1.
    thr = np.argwhere(dsq_bet == np.max(dsq_bet)).mean()  # Determine the threshold.
    img_bi = (img >= thr) * (nbins - 1)
    return img_bi


def adapt_otsu(img, size, kernel=[0, 0], nbins=256):
    loc_hist = local_histogram(img, size, kernel, nbins)
    img_bi = np.zeros_like(img)  # Initiate the binarized image.

    # Compute the threshold.
    while loc_hist.done == 0:   # Stop until all pixels are updated.
        cdf = np.cumsum(loc_hist.hist)
        npixels = cdf[-1]
        cdf = cdf / npixels  # Compute the cdf from 0 to L-1, and normalize it to interval [0,1].
        acm = np.cumsum(np.array(range(nbins)) * loc_hist.hist / npixels)  # Compute the accumulated average from 0 to L-1.
        mean_glb = acm[-1]  # The global mean.
        w1w2 = cdf * (1 - cdf)
        w1w2[w1w2 <= 1e-6] = 1  # Preprocess data to avoid the expression below being divided by 0.
        dsq_bet = (mean_glb * cdf - acm)**2 / w1w2  # Compute between-classes variance divided by threshold from 0 to L-1.
        thr = np.argwhere(dsq_bet == np.max(dsq_bet)).mean()  # Determine the threshold.
        img_bi[tuple(loc_hist.kernel)] = (img[tuple(loc_hist.kernel)] >= thr)*(nbins-1)  # Thresholding.
        loc_hist.update()  # Move to next pixel.
    return img_bi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process an image using local adaptive thresholding.')
    parser.add_argument('-i', '--img_source_path', type=str, required=True, help='the path of source image.')
    parser.add_argument('-s', '--size', type=int, required=True, help='a positive number n, yields an adjacent region \
                        with (2n+1)^2 pixels.')
    args = parser.parse_args()
    img = cv2.imread(args.img_source_path, 0)
    img_bi_glb = glb_otsu(img)
    img_bi_loc = adapt_otsu(img, size=args.size)

    plt.figure()

    plt.subplot(131)
    plt.title('Original')
    plt.imshow(img, cmap='gray')

    plt.subplot(132)
    plt.title('global_thresholding')
    plt.imshow(img_bi_glb, cmap='gray')

    plt.subplot(133)
    plt.title(f'{2*args.size + 1}x{2*args.size + 1} local_thresholding')
    plt.imshow(img_bi_loc, cmap='gray')

    plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, )
    plt.show()

    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_glb_bi.jpg', img_bi_glb)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_' + str(2*args.size+1) + 'x'
                + str(2*args.size+1) + '_bi.jpg', img_bi_loc)
