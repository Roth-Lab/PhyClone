import unittest

import numpy as np

from phyclone.tree.utils import _convolve_two_children, compute_log_S, compute_log_D
from phyclone.utils.cache import clear_convolution_caches
from phyclone.utils.math_utils import np_conv_dims


def non_cached_compute_log_S(child_log_R_values):
    """Compute log(S) recursion.

    Parameters
    ----------
    child_log_R_values: ndarray
        log_R values from child nodes.
    """
    if len(child_log_R_values) == 0:
        return 0.0

    log_D = compute_log_D(child_log_R_values)
    log_S = np.empty_like(log_D)
    log_S = np.logaddexp.accumulate(log_D, out=log_S, axis=-1)

    return np.ascontiguousarray(log_S)


class TestConvolutionCaching(unittest.TestCase):

    def __init__(self, method_name: str = ...):
        super().__init__(method_name)

        self.big_grid = 501
        self.default_grid_size = 101

        self.rng = np.random.default_rng(242643578967193853558243570818064774262)

    def setUp(self) -> None:
        clear_convolution_caches()
        self.eps = 1e-6

    def test_compute_log_S_no_children_no_hits(self):
        child_list = []

        expected = np.ascontiguousarray([0.0])
        actual = compute_log_S(child_list)

        num_hits = compute_log_S.cache_info().hits
        cache_size = compute_log_S.cache_info().currsize

        self.assertEqual(num_hits, 0)
        self.assertEqual(cache_size, 1)

        np.testing.assert_allclose(actual, expected)
        self.assertIsNot(actual, expected)

    def test_compute_log_S_no_children_multiple_hits(self):
        child_list = []

        num_cache_hits = 10

        for i in range(num_cache_hits):
            with self.subTest(msg="Num cache hits: {}".format(i), num_cache_hits=i):

                expected = np.ascontiguousarray([0.0])
                actual = compute_log_S(child_list)

                num_hits = compute_log_S.cache_info().hits
                cache_size = compute_log_S.cache_info().currsize

                self.assertEqual(num_hits, i)
                self.assertEqual(cache_size, 1)

                np.testing.assert_allclose(actual, expected)
                self.assertIsNot(actual, expected)

    def test_compute_log_S_1_child_no_hits(self):
        num_dims = 1000
        num_children = 1
        child_list = [np.log(self.rng.uniform(1e-6, 1.0, (num_dims, self.big_grid))) for _ in range(num_children)]

        expected = non_cached_compute_log_S(child_list)
        actual = compute_log_S(child_list)

        num_hits = compute_log_S.cache_info().hits
        cache_size = compute_log_S.cache_info().currsize

        self.assertEqual(num_hits, 0)
        self.assertEqual(cache_size, 1)

        np.testing.assert_allclose(actual, expected)
        self.assertIsNot(actual, expected)

    def test_compute_log_S_1_child_multiple_hits(self):
        num_dims = 1000
        num_children = 1
        child_list = [np.log(self.rng.uniform(1e-6, 1.0, (num_dims, self.big_grid))) for _ in range(num_children)]

        num_cache_hits = 10

        for i in range(num_cache_hits):
            with self.subTest(msg="Num cache hits: {}".format(i), num_cache_hits=i):

                expected = non_cached_compute_log_S(child_list)
                actual = compute_log_S(child_list)

                num_hits = compute_log_S.cache_info().hits
                cache_size = compute_log_S.cache_info().currsize

                self.assertEqual(num_hits, i)
                self.assertEqual(cache_size, 1)

                np.testing.assert_allclose(actual, expected)
                self.assertIsNot(actual, expected)

                self.rng.shuffle(child_list)

    def test_compute_log_S_10_children_no_hits(self):
        num_dims = 1000
        num_children = 10
        child_list = [np.log(self.rng.uniform(1e-6, 1.0, (num_dims, self.big_grid))) for _ in range(num_children)]

        expected = non_cached_compute_log_S(child_list)
        actual = compute_log_S(child_list)

        num_hits = compute_log_S.cache_info().hits
        cache_size = compute_log_S.cache_info().currsize

        self.assertEqual(num_hits, 0)
        self.assertEqual(cache_size, 1)

        np.testing.assert_allclose(actual, expected)
        self.assertIsNot(actual, expected)

    def test_compute_log_S_10_children_multiple_hits(self):
        num_dims = 1000
        num_children = 10
        child_list = [np.log(self.rng.uniform(1e-6, 1.0, (num_dims, self.big_grid))) for _ in range(num_children)]

        num_cache_hits = 10

        for i in range(num_cache_hits):
            with self.subTest(msg="Num cache hits: {}".format(i), num_cache_hits=i):

                expected = non_cached_compute_log_S(child_list)
                actual = compute_log_S(child_list)

                num_hits = compute_log_S.cache_info().hits
                cache_size = compute_log_S.cache_info().currsize

                self.assertEqual(num_hits, i)
                self.assertEqual(cache_size, 1)

                np.testing.assert_allclose(actual, expected)
                self.assertIsNot(actual, expected)
                self.rng.shuffle(child_list)

    def test_convolve_two_children_1_dim_no_hits(self):
        child_1 = self.rng.random(self.big_grid)
        child_2 = self.rng.random(self.big_grid)
        child_1_two_d = np.atleast_2d(child_1)
        child_2_two_d = np.atleast_2d(child_2)

        actual = _convolve_two_children(child_1_two_d, child_2_two_d)

        expected = np.atleast_2d(np.convolve(child_1, child_2)[: len(child_1)])

        num_hits = _convolve_two_children.cache_info().hits
        cache_size = _convolve_two_children.cache_info().currsize

        self.assertEqual(num_hits, 0)
        self.assertEqual(cache_size, 1)

        np.testing.assert_allclose(actual, expected)

    def test_convolve_two_children_1_dim_multiple_hits(self):
        child_1 = self.rng.random(self.big_grid)
        child_2 = self.rng.random(self.big_grid)
        child_1_two_d = np.atleast_2d(child_1)
        child_2_two_d = np.atleast_2d(child_2)

        num_cache_hits = 10

        for i in range(num_cache_hits):
            with self.subTest(msg="Num cache hits: {}".format(i), num_cache_hits=i):

                actual = _convolve_two_children(child_1_two_d.copy(), child_2_two_d.copy())

                expected = np.atleast_2d(np.convolve(child_1, child_2)[: len(child_1)])

                num_hits = _convolve_two_children.cache_info().hits
                cache_size = _convolve_two_children.cache_info().currsize

                self.assertEqual(num_hits, i)
                self.assertEqual(cache_size, 1)

                np.testing.assert_allclose(actual, expected)
                self.assertIsNot(actual, expected)

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
        self.assertIsNot(actual, expected)
        self.assertIsNot(actual_rev, expected)
        self.assertIs(actual, actual_rev)

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
        self.assertIsNot(actual, expected)
        self.assertIsNot(actual_rev, expected)
        self.assertIs(actual, actual_rev)

    def test_convolve_two_children_cache_order_1000_dims(self):
        num_dims = 1000
        child_1 = self.rng.random((num_dims, self.big_grid))
        child_2 = self.rng.random((num_dims, self.big_grid))

        actual = _convolve_two_children(child_1, child_2)
        actual_rev = _convolve_two_children(child_2, child_1)

        expected = np_conv_dims(child_1, child_2)

        num_hits = _convolve_two_children.cache_info().hits
        cache_size = _convolve_two_children.cache_info().currsize

        self.assertEqual(num_hits, 1)
        self.assertEqual(cache_size, 1)

        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actual_rev, actual)
        self.assertIsNot(actual, expected)
        self.assertIsNot(actual_rev, expected)
        self.assertIs(actual, actual_rev)

    def test_convolve_two_children_multiple_items_in_cache_1000_dims(self):
        num_dims = 1000
        num_items_in_cache = 10

        shape = (num_dims, self.big_grid)

        for i in range(num_items_in_cache):
            with self.subTest(msg="Num items in cache: {}".format(i + 1), items_in_cache=i + 1):

                child_1 = self._create_child_array(shape)
                child_2 = self._create_child_array(shape)

                actual = _convolve_two_children(child_1, child_2)
                actual_rev = _convolve_two_children(child_2, child_1)

                expected = np_conv_dims(child_1, child_2)
                expected_rev = np_conv_dims(child_2, child_1)

                num_hits = _convolve_two_children.cache_info().hits
                cache_size = _convolve_two_children.cache_info().currsize

                self.assertEqual(num_hits, i + 1)
                self.assertEqual(cache_size, i + 1)

                np.testing.assert_allclose(actual, expected)
                np.testing.assert_allclose(actual_rev, actual)
                np.testing.assert_allclose(actual_rev, expected_rev)
                np.testing.assert_allclose(expected_rev, actual_rev)
                self.assertIsNot(actual, expected)
                self.assertIsNot(actual_rev, expected)
                self.assertIsNot(actual_rev, expected_rev)
                self.assertIs(actual, actual_rev)

    def _create_child_array(self, shape):
        child_arr = self.rng.random(shape)
        child_arr.setflags(write=False)
        return child_arr


if __name__ == "__main__":
    unittest.main()
