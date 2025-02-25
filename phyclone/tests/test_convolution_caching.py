import unittest
import numpy as np
from phyclone.tree.utils import _convolve_two_children
from phyclone.utils.math import np_conv_dims
from phyclone.utils.dev import clear_convolution_caches



class TestConvolutionCaching(unittest.TestCase):

    def __init__(self, method_name: str = ...):
        super().__init__(method_name)

        self.big_grid = 1001
        self.default_grid_size = 101

        self.rng = np.random.default_rng(242643578967193853558243570818064774262)

    def setUp(self) -> None:
        clear_convolution_caches()

    def test_convolve_two_children_cache_order_1_dim(self):
        child_1 = self.rng.random(self.big_grid)
        child_2 = self.rng.random(self.big_grid)
        child_1_two_d = np.atleast_2d(child_1)
        child_2_two_d = np.atleast_2d(child_2)

        actual = _convolve_two_children(child_1_two_d, child_2_two_d)
        actual_rev = _convolve_two_children(child_2_two_d, child_1_two_d)

        expected = np.atleast_2d(np.convolve(child_1, child_2)[: len(child_1)])

        num_hits = _convolve_two_children.cache_info().hits
        cache_size = _convolve_two_children.cache_info().currsize

        self.assertEqual(num_hits, 1)
        self.assertEqual(cache_size, 1)

        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actual_rev, actual)

    def test_convolve_two_children_cache_order_3_dims(self):
        child_1 = self.rng.random((3, self.big_grid))
        child_2 = self.rng.random((3, self.big_grid))

        actual = _convolve_two_children(child_1, child_2)
        actual_rev = _convolve_two_children(child_2, child_1)

        expected = np_conv_dims(child_1, child_2)

        num_hits = _convolve_two_children.cache_info().hits
        cache_size = _convolve_two_children.cache_info().currsize

        self.assertEqual(num_hits, 1)
        self.assertEqual(cache_size, 1)

        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actual_rev, actual)


if __name__ == "__main__":
    unittest.main()