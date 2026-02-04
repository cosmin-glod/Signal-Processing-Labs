import numpy as np
from convolution import convolve2d

# Sobel kernels (3x3)
# These detect intensity changes in X and Y directions.
SOBEL_GX = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]], dtype=np.float32
)

SOBEL_GY = np.array(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]], dtype=np.float32
)


def gradient_sobel(image, padding="reflect"):
    """
    Compute Sobel gradients for a grayscale image

    Steps:
    1) Convolve with SOBEL_GX -> Ix (changes left/right)
    2) Convolve with SOBEL_GY -> Iy (changes up/down)
    3) Magnitude = sqrt(Ix^2 + Iy^2)  -> edge strength
    4) Angle = arctan2(Iy, Ix)        -> edge direction
    """

    # We only support 2D grayscale images here
    if image.ndim != 2:
        raise ValueError("gradient_sobel expects a 2D grayscale image")

    # 1) Horizontal changes (left-right)
    ix = convolve2d(image, SOBEL_GX, padding=padding)

    # 2) Vertical changes (up-down)
    iy = convolve2d(image, SOBEL_GY, padding=padding)

    # 3) Combine both directions into one "edge strength" image
    # np.hypot(ix, iy) <=> sqrt(ix**2 + iy**2)
    magnitude = np.hypot(ix, iy)

    # 4) Angle of the gradient at each pixel (in radians)
    angle = np.arctan2(iy, ix)

    return ix, iy, magnitude, angle
