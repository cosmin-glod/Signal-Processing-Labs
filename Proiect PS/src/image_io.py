from pathlib import Path
import cv2
import numpy as np


def load_image(path, grayscale=True):
    """
    Load an image and return float32 in [0, 1]
    If grayscale=True, returns 2D array
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {str(path)}")

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError(f"Failed to load image: {str(path)}")

    if grayscale:
        return (img.astype(np.float32) / 255.0).copy()

    # OpenCV reads BGR, convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0).copy()


def normalize_to_01(img):
    """Normalize any numeric array to [0, 1]"""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0

    img_f = img.astype(np.float32)
    min_val = float(np.min(img_f))
    max_val = float(np.max(img_f))

    if max_val <= 1.0 and min_val >= 0.0:
        return img_f

    if np.isclose(max_val, min_val):
        return np.zeros_like(img_f, dtype=np.float32)

    return (img_f - min_val) / (max_val - min_val)


def as_uint8(img01):
    """Convert [0,1] float image to uint8 0-255"""
    img_f = img01.astype(np.float32)
    img_f = np.clip(img_f, 0.0, 1.0)
    return (img_f * 255.0 + 0.5).astype(np.uint8)


def save_image(path, img):
    """Save an image to disk (float [0,1] or uint8)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if img.dtype == np.uint8:
        img8 = img
    else:
        img8 = as_uint8(normalize_to_01(img))

    # If color image in RGB, convert to BGR for OpenCV
    if img8.ndim == 3 and img8.shape[2] == 3:
        img8 = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)

    ok = cv2.imwrite(str(path), img8)
    if not ok:
        raise IOError(f"Failed to write image: {str(path)}")
