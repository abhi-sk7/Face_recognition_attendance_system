import cv2

import config


def rotate_frame(frame):
    rotations = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotation = config.CAMERA_ROTATE
    if rotation not in rotations:
        raise ValueError("CAMERA_ROTATE must be 0, 90, 180, or 270")
    code = rotations[rotation]
    return frame if code is None else cv2.rotate(frame, code)
