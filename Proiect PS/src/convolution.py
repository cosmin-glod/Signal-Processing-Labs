import numpy as np


def pad_image(image, pad_h, pad_w, mode):
    """Pad image with zeros or reflect"""
    if mode == "zero":
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    if mode == "reflect":
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    raise ValueError("Unsupported padding mode: " + str(mode))


def convolve2d(image, kernel, padding="reflect"):
    """Basic 2D convolution"""

    # Validate inputs
    if image.ndim != 2:
        raise ValueError("convolve2d expects a 2D grayscale image")
    if kernel.ndim != 2:
        raise ValueError("convolve2d expects a 2D kernel")

    # Get kernel size and validate odd dimensions
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel dimensions must be odd")

    # Compute padding sizes
    pad_h = kh // 2
    pad_w = kw // 2

    image_f = image.astype(np.float32, copy=False)
    kernel_f = kernel.astype(np.float32, copy=False)

    # Flip kernel for true convolution
    kernel_f = np.flipud(np.fliplr(kernel_f))

    padded = pad_image(image_f, pad_h, pad_w, padding)

    final = np.zeros_like(image_f, dtype=np.float32)
    for i in range(final.shape[0]):
        for j in range(final.shape[1]):
            zone = padded[i : i + kh, j : j + kw]
            final[i, j] = float(np.sum(zone * kernel_f))

    return final
