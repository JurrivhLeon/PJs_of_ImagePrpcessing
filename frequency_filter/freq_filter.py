import numpy as np


def dft(img):
    h, w = img.shape

    # Padding.
    img_pad = np.zeros((2 * h, 2 * w))
    img_pad[:h, :w] = img

    # Centralization.
    sign = -np.ones_like(img_pad)
    for i in range(2 * h):
        s = i % 2
        sign[i, s::2] = 1
    img_cen = img_pad * sign

    # DFT.
    return np.fft.fft2(img_cen)


def idft(img_freq):
    p, q = img_freq.shape
    # IDFT.
    img_re = np.fft.ifft2(img_freq)

    # Restore the sign.
    sign = -np.ones((p, q))
    for i in range(p):
        s = i % 2
        sign[i, s::2] = 1

    # Extract the filtered image from upper-left corner.
    img_re = np.real(img_re * sign)[:p // 2, :q // 2]
    return img_re.astype(int)


def symmetrize(f_mat):
    # Generate a centralized symmetric filter.
    h, w = f_mat.shape
    f_sym = np.zeros((2 * h, 2 * w))
    f_sym[:h, :w] = f_mat[::-1, ::-1]
    f_sym[:h, w:] = f_mat[::-1, :]
    f_sym[h:, :w] = f_mat[:, ::-1]
    f_sym[h:, w:] = f_mat
    return f_sym


def lowpass_ideal(p, q, radius=100):
    u_mat = np.array([np.arange(p // 2)] * (q // 2))
    v_mat = np.array([np.arange(q // 2)] * (p // 2))
    dist_sq = (u_mat ** 2).T + v_mat ** 2
    return symmetrize(dist_sq <= radius * radius)


def highpass_ideal(p, q, radius=100):
    return 1 - lowpass_ideal(p, q, radius)


def lowpass_gaussian(p, q, sd=100):
    u_mat = np.array([np.arange(p // 2)] * (q // 2))
    v_mat = np.array([np.arange(q // 2)] * (p // 2))
    dist_sq = (u_mat ** 2).T + v_mat ** 2
    return symmetrize(np.exp(-dist_sq / (2 * sd * sd)))


def highpass_gaussian(p, q, sd=100):
    return 1 - lowpass_gaussian(p, q, sd)


def laplacian(p, q):
    u_mat = np.array([np.arange(p // 2)] * (q // 2)) / p
    v_mat = np.array([np.arange(q // 2)] * (p // 2)) / q
    dist_sq = (u_mat ** 2).T + v_mat ** 2
    return - symmetrize(4 * np.pi**2 * dist_sq)


