import numpy as np
import os.path as osp
import cv2
import argparse
import matplotlib.pyplot as plt


def lin_interp(img, t):
    h, w = img.shape
    img_interp = np.zeros(((h - 1) * t + 1, (w - 1) * t + 1))  # Initiate the interpolated image.
    weight = [i / t for i in range(t)]  # Initiate the weight of interpolation.
    r = np.array([weight * (h - 1)])  # The weight of each row in the image.
    s = np.array([weight * (w - 1)])  # The weight of each column in the image.
    block = np.ones((t, t))  # A block in the image matrix.
    lblock = np.ones((1, t))  # A line block in the edge of the image matrix.

    # Process pixels in the image.
    # The weight of the nearest pixels:
    # upper-left: (1-r)*(1-s), upper-right: r*(1-s), lower-left: (1-r)*s, lower-right: r*s
    img_interp[:(h-1) * t, :(w-1) * t] = (
            np.kron(img[:h-1, :w-1], block) * ((1 - r).T * (1 - s))
            + np.kron(img[1:, :w-1], block) * (r.T * (1 - s))
            + np.kron(img[:h-1, 1:], block) * ((1 - r).T * s)
            + np.kron(img[1:, 1:], block) * (r.T * s)
    )

    # Supplement pixels in the edge of the image.
    img_interp[:(h-1) * t, (w-1) * t] = np.kron(img[:h-1, w-1], lblock) * (1 - r) + np.kron(img[1:, w-1], lblock) * r
    img_interp[(h-1) * t, :(w-1) * t] = np.kron(img[h-1, :w-1], lblock) * (1 - s) + np.kron(img[h-1, 1:], lblock) * s
    img_interp[(h-1) * t, (w-1) * t] = img[h-1, w-1]

    '''
    # Process by loop.
    for x in range(h-1):
        r = np.array([[i/t for i in range(t)]])
        for y in range(w-1):
            img_interp[t*x:t*(x+1), t*y:t*(y+1)] = img[x, y]*(1-r).T*(1-r) + img[x+1, y]*r.T*(1-r) \
                                                   + img[x, y+1]*(1-r).T*r + img[x+1, y+1]*r.T*r
        img_interp[t*x:t*(x+1), t*(w-1)] = img[x, w-1]*(1-r) + img[x+1, w-1]*r

    for y in range(w-1):
        s = np.array([[j/t for j in range(t)]])
        img_interp[t*(h-1), t*y:t*(y+1)] = img[h-1, y]*(1-s) + img[h-1, y+1]*s

    img_interp[t*(h-1), t*(w-1)] = img[h-1, w-1]'''

    return img_interp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process an image using linear interpolation.')
    parser.add_argument('-i', '--img_source_path', type=str, required=True, help='the source path of pending image.')
    parser.add_argument('-t', '--times', type=int, required=True, help='a positive integer t, the times of zoom-in.')
    args = parser.parse_args()
    img = cv2.imread(args.img_source_path, 0)
    img_interp = lin_interp(img, args.times)

    plt.figure()

    plt.subplot(121)
    plt.title('Original')
    plt.imshow(img, cmap='gray')

    plt.subplot(122)
    plt.title(f'{args.times} times interpolation')
    plt.imshow(img_interp, cmap='gray')

    plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, )
    plt.show()

    cv2.imwrite(osp.splitext(args.img_source_path)[0] + '_' + str(args.times) + 'x' +
                '_interp.jpg', img_interp)
