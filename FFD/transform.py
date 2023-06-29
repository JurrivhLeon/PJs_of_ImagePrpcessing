import os.path as osp
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from ffd import FFD


def generate_coord(shape):
    """
    Generate the coordinates of pixels with a given shape.
    Input: the shape of image (h, w), height and width;
    Output: an array of dimension 2 * N, N = h * w, with each column the coordinate of a pixel.
    """
    h = np.linspace(0, shape[0] - 1, shape[0]).astype(float)
    w = np.linspace(0, shape[1] - 1, shape[1]).astype(float)
    hh, ww = np.meshgrid(h, w, indexing='ij')
    coord = np.stack((hh, ww), axis=-1)
    return coord.reshape((-1, 2)).T


def bilinear_interp(img: np.ndarray, coord: np.ndarray):
    """
    Based on inverse transformation, compute pixel values.
    Input: image with shape h * w, transformed coordinate matrix with shape 2 * hw;
    Output: transformed image with shape
    """
    h, w = img.shape
    coord_floor = np.floor(coord).astype(int)
    offset = coord - coord_floor
    get_pixel = lambda a, b: img[a.clip(0, h - 1), b.clip(0, w - 1)]
    la = (1 - offset[0, :]) * (1 - offset[1, :]) * get_pixel(coord_floor[0, :], coord_floor[1, :])
    lb = offset[0, :] * (1 - offset[1, :]) * get_pixel(coord_floor[0, :] + 1, coord_floor[1, :])
    ra = (1 - offset[0, :]) * offset[1, :] * get_pixel(coord_floor[0, :], coord_floor[1, :] + 1)
    rb = offset[0, :] * offset[1, :] * get_pixel(coord_floor[0, :] + 1, coord_floor[1, :] + 1)
    return (la + lb + ra + rb).reshape((h, w))


def main():
    parser = argparse.ArgumentParser(description='Transformation')
    parser.add_argument('-i', '--img_path', type=str, default='images/cat.png',
                        help='the path of source image.')
    parser.add_argument('-o', '--offset_path', type=str, default='data/trans1.txt',
                        help='the path of txt file which stores the offset matrix of control points.')
    args = parser.parse_args()

    img = cv2.imread(args.img_path)
    if img.shape[0] != 513 or img.shape[1] != 513:
        img = cv2.resize(img, (513, 513), interpolation=cv2.INTER_AREA)
    r, g, b = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    coord = generate_coord(r.shape)
    npy_path = osp.splitext(args.offset_path)[0] + '_inv.npy'

    if osp.exists(npy_path):
        # If the data has been fitted, use the saved result directly.
        ffdinv = np.load(npy_path, allow_pickle=True).item()

    else:
        transformer = FFD()
        transformer.update_offset(args.offset_path)
        ffdinv = transformer.FFDinvtransform(coord)
        # Save the inverse transformation result.
        np.save(npy_path, ffdinv)

    print('Restore the image by interpolation.')
    img_ffd_r = bilinear_interp(r, ffdinv['inv'])
    img_ffd_g = bilinear_interp(g, ffdinv['inv'])
    img_ffd_b = bilinear_interp(b, ffdinv['inv'])
    img_ffd = np.zeros_like(img)
    img_ffd[:, :, 2], img_ffd[:, :, 1], img_ffd[:, :, 0] = img_ffd_r, img_ffd_g, img_ffd_b
    print('mse: {0}'.format(ffdinv['mse']))

    # Visualization.
    plt.figure()

    plt.subplot(121)
    plt.title('Original')
    plt.imshow(img[:, :, ::-1])
    plt.axis('off')

    plt.subplot(122)
    plt.title('Transformed')
    plt.imshow(img_ffd[:, :, ::-1])
    plt.axis('off')

    plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, )
    plt.show()

    tr = osp.split(osp.splitext(args.offset_path)[0])[-1]
    cv2.imwrite(osp.splitext(args.img_path)[0] + f'_ffd_{tr}.png', img_ffd)


if __name__ == '__main__':
    main()
