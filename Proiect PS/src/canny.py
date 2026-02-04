import numpy as np

import convolution
import sobel
import image_io


def gaussian_kernel(k_size, sigma):
    """
    Create a 2D Gaussian kernel
    k_size must be odd
    """
    if k_size <= 0:
        raise ValueError("k_size must be > 0")
    if k_size % 2 == 0:
        raise ValueError("k_size must be odd")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    # this is used to center the kernel
    half = k_size // 2

    # Empty kernel
    kernel = np.zeros((k_size, k_size), dtype=np.float32)

    # Fill the kernel using the Gaussian formula
    total = 0.0
    for i in range(k_size):
        for j in range(k_size):
            x = i - half
            y = j - half
            value = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
            kernel[i, j] = value
            total += value

    # Normalize
    if total != 0.0:
        kernel = kernel / total

    return kernel


def gaussian_blur(image, k_size, sigma):
    """Apply Gaussian blur using our own convolution"""
    if k_size <= 0:
        return image
    kernel = gaussian_kernel(k_size, sigma)
    return convolution.convolve2d(image, kernel, padding="reflect")


def non_max_suppression(magnitude, angle):
    """
    Thin edges by keeping only local maxima along the gradient direction
    Angle is in radians
    """
    h, w = magnitude.shape
    out = np.zeros_like(magnitude, dtype=np.float32)

    # Convert angle to degrees in [0, 180)
    angle_deg = np.degrees(angle)
    angle_deg[angle_deg < 0] += 180.0

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle_deg[i, j]

            # Choose two neighbors along the gradient direction
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1 = magnitude[i, j - 1]
                n2 = magnitude[i, j + 1]
            elif 22.5 <= a < 67.5:
                n1 = magnitude[i + 1, j - 1]
                n2 = magnitude[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                n1 = magnitude[i - 1, j]
                n2 = magnitude[i + 1, j]
            else:
                n1 = magnitude[i - 1, j - 1]
                n2 = magnitude[i + 1, j + 1]

            # Keep only local maxima
            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                out[i, j] = magnitude[i, j]
            else:
                out[i, j] = 0.0

    return out


def double_threshold(magnitude, low, high):
    """Classify pixels as strong, weak, or non-edge"""
    strong = magnitude >= high
    weak = (magnitude >= low) & ~strong
    return strong, weak


def hysteresis(strong, weak):
    """keep weak pixels connected to strong pixels"""
    h, w = strong.shape
    out = np.zeros((h, w), dtype=np.uint8)

    # Start with all strong edges
    out[strong] = 255

    # keep adding weak pixels that touch strong pixels
    changed = True
    while changed:
        changed = False
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if weak[i, j] and out[i, j] == 0:
                    # If any neighbor is strong, promote this weak pixel
                    if np.any(out[i - 1:i + 2, j - 1:j + 2] == 255):
                        out[i, j] = 255
                        changed = True

    return out


def canny_edges(image01, low, high, blur_k_size=0, blur_sigma=1.0):
    """
    Input image must be grayscale float in [0,1]
    Low/high can be in [0,1] or [0,255]
    """
    if image01.ndim != 2:
        raise ValueError("canny_edges expects a 2D grayscale image")

    # Normalize image just in case
    img = image_io.normalize_to_01(image01)

    # Blur to reduce noise
    if blur_k_size > 0:
        img = gaussian_blur(img, blur_k_size, blur_sigma)

    # Gradients from Sobel
    ix, iy, mag, ang = sobel.gradient_sobel(img, padding="reflect")

    mag = image_io.normalize_to_01(mag)
    mag_thin = non_max_suppression(mag, ang)

    # Convert thresholds to [0,1] if needed
    low_f = float(low)
    high_f = float(high)
    if low_f > 1.0 or high_f > 1.0:
        low_f = low_f / 255.0
        high_f = high_f / 255.0

    strong, weak = double_threshold(mag_thin, low_f, high_f)

    edges = hysteresis(strong, weak)

    return edges
