import unittest
import numpy as np
import numpy.testing as npt
from src.direct_linear_transform import direct_linear_transform


class TestDLT(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.img_pts = np.array(
            [
                [-0.22895, 0.45549],
                [-2.17382, 0.74228],
                [-3.33303, 2.73776],
                [-1.42431, 2.68233],
                [5.73098, -0.51333],
                [-5.77195, -0.34945],
                [1.97138, -1.55927],
                [1.24597, 0.23136],
                [-1.04835, -3.45594],
                [4.8923, -0.13014],
                [2.92092, 0.95677],
                [-0.62419, 2.61003],
            ]
        )
        obj_pts = np.loadtxt("data/passpunkte.txt", delimiter=";", skiprows=1)
        cls.obj_pts = obj_pts[:, 1:4]

        cls.r_mat = np.array(
            [
                [-0.80183517, -0.42505956, 0.42138475],
                [0.59754298, -0.5655176, 0.56741293],
                [-0.00165737, 0.7067632, 0.70744432],
            ]
        )
        cls.t_vec = np.array([[64.33942777], [227.037899], [112.7420435]])
        cls.interior_orientation = [
            -0.039917370836011955,
            -0.04877527504755159,
            8.851666118124498,
            8.83333418875823,
        ]

        return super().setUpClass()

    def test_returned_type(self):
        returned_types = [np.ndarray, np.ndarray, list]
        result = direct_linear_transform(self.img_pts, self.obj_pts)
        for i, res in enumerate(result):
            self.assertIsInstance(res, returned_types[i])

    def test_results(self):
        res = direct_linear_transform(self.img_pts, self.obj_pts)
        test_array = np.vstack((np.hstack((res[0], res[1])), np.array([res[2]])))
        npt.assert_allclose(
            test_array,
            np.vstack((np.hstack((self.r_mat, self.t_vec)), np.array([self.interior_orientation]))),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
