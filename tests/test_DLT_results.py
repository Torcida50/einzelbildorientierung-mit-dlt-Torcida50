import unittest
import numpy as np
import numpy.testing as npt
from src.utils import rotation_matrix_from_DLT


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
        cls.testdata = [-71.22, np.array([ 5.40228962e-01,  5.35106178e-03,  1.52789137e-02, -3.19032849e+00,
                                        -3.05283293e-03,  4.19004076e-01,  3.46582310e-01, -3.57419780e+00,
                                        -1.33780294e-03,  9.86409887e-03, -9.90283271e-03]), -4.16, 3.58, 38.26, 38.56]
        return super().setUpClass()

    def test_orthogonality(self):
        rot_mat = rotation_matrix_from_DLT(self.testdata[0],self.testdata[1],self.testdata[2],self.testdata[3],self.testdata[4],self.testdata[5])[0]
        npt.assert_allclose(rot_mat.T @ rot_mat, np.eye(3), atol=0.1, rtol=1e-03)


if __name__ == "__main__":
    unittest.main()