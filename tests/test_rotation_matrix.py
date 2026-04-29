import sys
import unittest
import numpy as np
import numpy.testing as npt

from src.rotation_matrix import rotation_matrix


class TestRotationMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.r_mat = np.array(
            [
                [0.77837848, -0.37964055, 0.5],
                [0.58722257, 0.72195222, -0.36599815],
                [-0.22202837, 0.57849637, 0.78488557],
            ]
        )
        cls.euler_deg = [25, 30, 26]
        cls.euler_rad = [0.43633231, 0.52359878, 0.45378561]
        return super().setUpClass()

    def test_returned_type(self):
        self.assertTrue(isinstance(rotation_matrix(*self.euler_rad), np.ndarray))

    def test_rotation_matrix_rad(self):
        npt.assert_allclose(rotation_matrix(*self.euler_rad), self.r_mat)

    def test_rotation_matrix_deg(self):
        npt.assert_allclose(rotation_matrix(*self.euler_deg, 360), self.r_mat)

    def test_orthogonality(self):
        rot_mat = rotation_matrix(*self.euler_rad)
        npt.assert_allclose(rot_mat.T @ rot_mat, np.eye(3), atol=1e-07)


if __name__ == "__main__":
    unittest.main()
