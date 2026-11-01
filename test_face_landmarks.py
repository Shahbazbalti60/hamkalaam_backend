# -*- coding: utf-8 -*-
"""Run:  python test_face_landmarks.py"""
import unittest
import numpy as np
from app.ai import face_landmarks as fl


def fake_face(seed=0):
    """A rough synthetic 468-point 'face' in pixels (only the 68 used points matter)."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 3, (468, 2)).astype(np.float32) + 200
    pts[33] = (150, 200); pts[263] = (250, 200)      # outer eye corners, 100 px apart
    pts[fl.LIPS[0]] = (170, 280); pts[fl.LIPS[10]] = (230, 280)
    return pts


class FaceLandmarkTests(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(fl.N_POINTS, 68)
        self.assertEqual(fl.VECTOR_LEN, 136)
        self.assertEqual(len(set(fl.INDICES)), 68)
        self.assertTrue(all(0 <= i < 468 for i in fl.INDICES))
        self.assertEqual(len(fl.RIGHT_BROW + fl.LEFT_BROW), 20)
        self.assertEqual(len(fl.LIPS), 20)
        self.assertEqual(len(fl.NOSE_BRIDGE), 8)

    def test_select_shape(self):
        sel = fl.select_points(fake_face())
        self.assertEqual(sel.shape, (68, 2))

    def test_same_after_move_scale_rotate(self):
        base = fl.select_points(fake_face(1))
        ref = fl.normalize_face(base)
        for scale, angle, shift in [(0.5, 0.0, (10, 20)), (2.3, 0.3, (-50, 80)), (1.0, -0.4, (0, 0))]:
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]])
            moved = (base @ rot.T) * scale + np.array(shift)
            np.testing.assert_allclose(fl.normalize_face(moved), ref, atol=1e-3)

    def test_expression_changes_the_vector(self):
        base = fl.select_points(fake_face(2))
        opened = base.copy()
        for i, idx in enumerate(fl.INDICES):
            if idx in (17, 14, 84, 314, 405, 321, 181, 91):      # lower lip
                opened[i, 1] += 25
        a, b = fl.normalize_face(base), fl.normalize_face(opened)
        self.assertGreater(float(np.abs(a - b).max()), 0.1)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            fl.normalize_face(np.full((68, 2), np.nan))
        with self.assertRaises(ValueError):
            fl.normalize_face(np.zeros((68, 2)))
        with self.assertRaises(Exception):
            fl.normalize_face(np.zeros((10, 2)))

    def test_flat_vector(self):
        sel = fl.select_points(fake_face(3))
        self.assertEqual(fl.vector_to_input(sel.reshape(-1)).shape, (68, 2))


if __name__ == "__main__":
    unittest.main(verbosity=1)
