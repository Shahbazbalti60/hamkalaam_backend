# -*- coding: utf-8 -*-
"""
app/ai/face_landmarks.py   (SRDS FR-20)

Picks 68 of the 468 MediaPipe FaceMesh landmarks - the four regions that
carry grammar and emotion in PSL non-manual markers - and turns them
into the 136-number vector the expression CNN uses.

Regions (68 points): eyebrows 20, eyes 20, mouth/lips 20, nose bridge 8.
(The SRDS text splits 24/16/20/8; the same four regions and the same
total of 68 are used here. The indices were checked on a real face:
brows above eyes above nose above lips.)

The SAME file is used by the training notebook and by the server, so
training data and live data are always prepared identically.

Input  : 68 (x, y) points in PIXELS (x = fraction * image width,
         y = fraction * image height), in the order of INDICES.
Output : 68 x 2 floats, moved to the face centre, rotated so the eyes
         are level and scaled by the distance between the outer eye
         corners -> independent of position, distance and head tilt.
"""
import numpy as np

# MediaPipe FaceMesh indices ("right/left" = the person's own side)
RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_EYE = [33, 160, 159, 158, 133, 153, 145, 144, 163, 7]
LEFT_EYE = [362, 385, 386, 387, 263, 249, 390, 373, 374, 380]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1, 19]

INDICES = RIGHT_BROW + LEFT_BROW + RIGHT_EYE + LEFT_EYE + LIPS + NOSE_BRIDGE
N_POINTS = len(INDICES)            # 68
VECTOR_LEN = N_POINTS * 2          # 136

_OUTER_RIGHT_EYE = INDICES.index(33)
_OUTER_LEFT_EYE = INDICES.index(263)

assert N_POINTS == 68 and len(set(INDICES)) == 68


def select_points(all_points):
    """468 (or 478) x 2 array of pixel coordinates -> 68 x 2."""
    arr = np.asarray(all_points, dtype=np.float32)
    return arr[INDICES, :2]


def normalize_face(points68):
    """68 x 2 pixel points (or a flat list of 136) -> 68 x 2 normalised."""
    pts = np.asarray(points68, dtype=np.float32).reshape(N_POINTS, 2)
    if not np.isfinite(pts).all():
        raise ValueError("face points contain invalid numbers")

    pts = pts - pts.mean(axis=0, keepdims=True)

    # roll: rotate so the line between the outer eye corners is horizontal
    eye_vec = pts[_OUTER_LEFT_EYE] - pts[_OUTER_RIGHT_EYE]
    width = float(np.linalg.norm(eye_vec))
    if width < 1e-6:
        raise ValueError("face is too small / degenerate")
    angle = np.arctan2(eye_vec[1], eye_vec[0])
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = pts @ rot.T

    return (pts / width).astype(np.float32)


def vector_to_input(vec136):
    """136 raw pixel numbers -> normalised 68 x 2 model input."""
    return normalize_face(np.asarray(vec136, dtype=np.float32).reshape(N_POINTS, 2))
