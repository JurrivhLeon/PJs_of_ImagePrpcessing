import os.path as osp
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from filter import lin_filter, normalize


def sharpen_laplacian(img, w, nbins=256):
    # return a sharpened image f + w * D2f.
    laplacian_kernel = np.array([[1.0, 1.0, 1.0],
                                 [1.0, -8.0, 1.0],
                                 [1.0, 1.0, 1.0]]
                                )
    laplacian_filter = lin_filter(img.astype(float), laplacian_kernel)
    img_sharpened_laplacian = img.astype(float)
    # The boundary of filtered image tends to have small values.
    # We remain the boundary intact to avoid aberrant bright boundary.
    img_sharpened_laplacian[1:-1, 1:-1] -= w * laplacian_filter[1:-1, 1:-1]
    img_sharpened_laplacian[img_sharpened_laplacian < 0] = 0
    img_sharpened_laplacian[img_sharpened_laplacian > nbins - 1] = nbins - 1
    return normalize(laplacian_filter), img_sharpened_laplacian.astype(int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smooth an image.')
    parser.add_argument('-i', '--img_source_path', type=str, required=True,
                        help='the path of source image.')
    parser.add_argument('-w', '--laplacian_weight', type=float, default=1.0,
                        help='the sharpening weight of laplacian filter.')
    args = parser.parse_args()

    img = cv2.imread(args.img_source_path, 0).astype(float)
    filtering, laplace = sharpen_laplacian(img, args.laplacian_weight)

    plt.figure()

    plt.subplot(221)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(222)
    plt.title('Laplacian filter')
    plt.imshow(filtering, cmap='gray')
    plt.axis('off')

    plt.subplot(224)
    plt.title('Laplacian Sharpened')
    plt.imshow(laplace, cmap='gray')
    plt.axis('off')

    plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, hspace=0.35)
    plt.show()

    cv2.imwrite(args.img_source_path, img)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + f'_laplacian_filter.jpg', filtering)
    cv2.imwrite(osp.splitext(args.img_source_path)[0] + f'_{args.laplacian_weight}_laplacian.jpg', laplace)
