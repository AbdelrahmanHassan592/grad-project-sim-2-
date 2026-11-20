"""Estimate a course beacon's bearing and depth from the simulated RGB camera."""
from math import tan

import numpy as np

# Keep these dimensions aligned with the course beacon and robot camera model.
BEACON_HEIGHT = 0.60
BEACON_RADIUS = 0.095
CAMERA_OFFSET_X = 0.31
CAMERA_HORIZONTAL_FOV = 1.047


def beacon_offset(image, name):
    """Return forward/left position relative to base_link, or None."""
    if image.encoding.lower() not in ('rgb8', 'bgr8') or image.width == 0 or image.height == 0:
        return None
    stride = image.step
    if stride < image.width * 3 or len(image.data) != stride * image.height:
        return None
    rgb = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, stride)
    rgb = rgb[:, :image.width * 3].reshape(image.height, image.width, 3)
    r, g, b = np.moveaxis(rgb, 2, 0)
    if image.encoding.lower() == 'bgr8':
        r, b = b, r
    if name == 'B':
        mask = (b > 170) & (r < 85) & (g < 130)
    elif name == 'C':
        mask = (r > 190) & (g < 95) & (b < 75)
    else:
        return None
    rows, cols = np.nonzero(mask)
    if rows.size < 60:
        return None
    top, bottom = np.percentile(rows, (1, 99))
    left, right = np.percentile(cols, (1, 99))
    height, width = bottom - top + 1, right - left + 1
    if not (top > 2 and bottom < image.height - 3 and left > 2 and right < image.width - 3):
        return None
    if not (height >= 18 and 1.6 <= height / width <= 4.8 and rows.size >= .50 * height * width):
        return None
    focal_length = (image.width / 2) / tan(CAMERA_HORIZONTAL_FOV / 2)
    # Percentile bounds omit the outer 2% of the pixels.
    depth = focal_length * BEACON_HEIGHT * .98 / height
    if not (0.75 <= depth <= 7.5):
        return None
    left_offset = (image.width / 2 - (left + right) / 2) * depth / focal_length
    return CAMERA_OFFSET_X + depth + BEACON_RADIUS, left_offset
