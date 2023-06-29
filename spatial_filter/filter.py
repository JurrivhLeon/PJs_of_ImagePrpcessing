import numpy as np

# Pad the bound of an image with pixel 0 so that the filter can deal with pixels on the bound.
def padding(img, pad):
    h, w = img.shape
    img_pad = np.zeros((h + 2 * pad, w + 2 * pad))
    img_pad[pad:h+pad, pad:w+pad] = img
    return img_pad

# Compute the inner product of filter and sub-image.
def inner_prod(x, y):
    if not x.shape == y.shape:
        raise ValueError
    return np.sum(x * y)

# Linear filter, in form of spatial-correlation.
def lin_filter(img, kernel):
    h, w = img.shape
    img_filter = np.zeros_like(img)
    edge = kernel.shape[0]
    img_pad = padding(img, edge // 2)
    # Process each pixel.
    for i in range(h):
        for j in range(w):
            img_filter[i, j] = inner_prod(img_pad[i:i+edge, j:j+edge], kernel)
    return img_filter

# Median filter.
def median_filter(img, s):
    h, w = img.shape
    img_filter = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            # determine the bound of sub-image covered by the filter.
            a, b, l, r = max(0, i-s), min(h, i+s+1), max(0, j-s), min(w, j+s+1)
            img_filter[i, j] = np.median(img[a:b, l:r].reshape((b-a) * (r-l)))
    return img_filter

# Normalize the filtered image onto [0, nbins-1].
def normalize(img, nbins=256):
    # In case some pixel values of processed image overflow or underflow on [0, nbins - 1],
    # normalize it to the interval [0, nbins - 1].
    lb = np.min(img)
    ub = np.max(img)
    img_normalized = (img - lb) / (ub - lb) * (nbins - 1)
    return img_normalized.astype(int)
