import unittest
import numpy as np
import numpy.testing as npt

from src.rotation_matrix import rotation_matrix
from src.collinearity_equation import (
    auxilary_terms_coll_equation,
    collinearity_equation,
    part_derivatives_coll_equation,
)


class TestAuxTermCollEquation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aux_terms = [-49.0, -82.72440929664947, -11.907649076136458]
        cls.object_coord = [1.0, 4.4, 7.6]
        cls.proj_center_pos = [50, 60, 70]
        cls.proj_center_ori = [0.7, 0, 0]
        return super().setUpClass()

    def test_returned_type(self):
        self.assertTrue(
            isinstance(
                auxilary_terms_coll_equation(
                    *self.object_coord,
                    *self.proj_center_pos,
                    rotation_matrix(*self.proj_center_ori)
                ),
                tuple,
            )
        )

    def test_auxilary_terms_coll_equation(self):
        npt.assert_allclose(
            auxilary_terms_coll_equation(
                *self.object_coord, *self.proj_center_pos, rotation_matrix(*self.proj_center_ori)
            ),
            self.aux_terms,
        )


class TestCollEquation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image_coord = [-156.37007675440688, -263.99229043224585]
        cls.object_coord = [1.0, 4.4, 7.6]
        cls.proj_center_pos = [50, 60, 70]
        cls.proj_center_ori = [0.7, 0, 0]
        cls.principal_point = [0, 0]
        cls.pd = 38
        return super().setUpClass()

    def test_returned_type(self):
        self.assertTrue(
            isinstance(
                collinearity_equation(
                    *self.object_coord,
                    *self.proj_center_pos,
                    *self.principal_point,
                    self.pd,
                    rotation_matrix(*self.proj_center_ori)
                ),
                tuple,
            )
        )

    def test_collinearity_equation(self):
        npt.assert_allclose(
            collinearity_equation(
                *self.object_coord,
                *self.proj_center_pos,
                *self.principal_point,
                self.pd,
                rotation_matrix(*self.proj_center_ori)
            ),
            self.image_coord,
        )


class TestPartDerivatCollEquation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.part_dervatives = [
            -3.1912260562123844,
            -8.459803321025397,
            10.04383239428556,
            -1086.3288083542618,
            681.4631816889236,
            -263.99229043224585,
            0.0,
            -16.723074776088637,
            14.900688422284107,
            -1871.9981423069275,
            -1086.328808354262,
            156.37007675440685,
        ]
        cls.object_coord = [1.0, 4.4, 7.6]
        cls.proj_center_pos = [50, 60, 70]
        cls.proj_center_ori = [0.7, 0, 0]
        cls.principal_point = [0, 0]
        cls.pd = 38
        return super().setUpClass()

    def test_returned_type(self):
        self.assertTrue(
            isinstance(
                part_derivatives_coll_equation(
                    *self.object_coord,
                    *self.proj_center_pos,
                    self.pd,
                    self.proj_center_ori[2],
                    rotation_matrix(*self.proj_center_ori)
                ),
                tuple,
            )
        )

    def test_returned_num_elements(self):
        self.assertEqual(
            len(
                part_derivatives_coll_equation(
                    *self.object_coord,
                    *self.proj_center_pos,
                    self.pd,
                    self.proj_center_ori[2],
                    rotation_matrix(*self.proj_center_ori)
                )
            ),
            12,
        )

    def test_part_derivatives_coll_equation(self):
        npt.assert_allclose(
            part_derivatives_coll_equation(
                *self.object_coord,
                *self.proj_center_pos,
                self.pd,
                self.proj_center_ori[2],
                rotation_matrix(*self.proj_center_ori)
            ),
            self.part_dervatives,
        )


if __name__ == "__main__":
    unittest.main()
