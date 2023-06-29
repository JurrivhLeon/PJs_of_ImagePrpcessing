import os.path as osp
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from filter import normalize
from smooth import smooth_gaussian

def highboost(img, k, nbins=256):
    # return a sharpened image f + k * (f - f_bar).
    img_mean = smooth_gaussian(img, sigma_sq=9.0, s=3)
    mask = (img - img_mean)
    img_hboosted = img + k * mask
    img_hboosted[img_hboosted < 0] = 0
    img_hboosted[img_hboosted > nbins - 1] = nbins - 1
    return normalize(mask), img_hboosted.astype(int)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Smooth an image.')
    parser.add_argument('-i', '--img_source_path', type=str, required=True,
                        help='the path of source image.')
    parser.add_argument('-k', '--highboost_weight', type=float, default=1.0,
                        help='use highboost')
    args = parser.parse_args()

    img = cv2.imread(args.img_source_path, 0).astype(float)
    mask, img_hb = highboost(img, args.highboost_weight)

    plt.subplot(221)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(222)
    plt.title('Highboost mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(224)
    plt.title('Highboost sharpened')
    plt.imshow(img_hb, cmap='gray')
    plt.axis('off')

    plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, hspace=0.35)
    plt.show()

    cv2.imwrite(args.img_source_path, img)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + f'_highboost_mask.jpg', mask)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + f'_{args.highboost_weight}_highboost.jpg', img_hb)