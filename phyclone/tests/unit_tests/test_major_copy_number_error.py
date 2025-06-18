import unittest
from phyclone.utils.exceptions import MajorCopyNumberError
from phyclone.data.pyclone import get_major_cn_prior


class TestMajorCopyNumberError(unittest.TestCase):

    def test_major_less_than_minor(self):
        with self.assertRaises(MajorCopyNumberError):
            get_major_cn_prior(0, 1, 2)

    def test_major_and_minor_cn_equal(self):
        maj = 1
        minor = 1
        try:
            get_major_cn_prior(maj, minor, 2)
        except MajorCopyNumberError:
            self.fail("MajorCopyNumberError raised with major: {}, and minor: {}".format(maj, minor))

    def test_major_greater_than_minor(self):
        maj = 2
        minor = 1
        try:
            get_major_cn_prior(maj, minor, 2)
        except MajorCopyNumberError:
            self.fail("MajorCopyNumberError raised with major: {}, and minor: {}".format(maj, minor))


if __name__ == "__main__":
    unittest.main()
