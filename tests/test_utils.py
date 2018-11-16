import unittest
import math
import cmath
from pyvvo import utils


class TestParseComplexStr(unittest.TestCase):
    """Test utils.parse_complex_str.

    Unfortunately these tests aren't going to have very informative
    names.
    """

    def test_polar_1(self):
        num, unit = utils.parse_complex_str('+348863+13.716d VA')
        self.assertEqual(unit, 'VA')
        expected_num = 348863 * cmath.exp(1j * math.radians(13.716))
        self.assertEqual(num, expected_num)

    def test_polar_2(self):
        num, unit = utils.parse_complex_str('-12.2+13d I')
        self.assertEqual(unit, 'I')
        expected_num = -12.2 * cmath.exp(1j * math.radians(13))
        self.assertEqual(num, expected_num)

    def test_polar_3(self):
        num, unit = utils.parse_complex_str('+3.258-2.14890r kV')
        self.assertEqual(unit, 'kV')
        expected_num = 3.258 * cmath.exp(1j * -2.14890)
        self.assertEqual(num, expected_num)

    def test_polar_4(self):
        num, unit = utils.parse_complex_str('-1.5e02+12d f')
        self.assertEqual(unit, 'f')
        expected_num = -1.5e02 * cmath.exp(1j * math.radians(12))
        self.assertEqual(num, expected_num)

    def test_rect_1(self):
        num, unit = utils.parse_complex_str('-1+2j VAr')
        self.assertEqual(unit, 'VAr')
        expected_num = -1 + 1j * 2
        self.assertEqual(num, expected_num)

    def test_rect_2(self):
        num, unit = utils.parse_complex_str('+1.2e-003+1.8e-2j d')
        self.assertEqual(unit, 'd')
        expected_num = 1.2e-003 + 1j * 1.8e-2
        self.assertEqual(num, expected_num)

    def test_non_complex_num(self):
        self.assertRaises(ValueError, utils.parse_complex_str, '15')

    def test_weird_string(self):
        self.assertRaises(ValueError, utils.parse_complex_str,
                          'Look mom, a string!')

    def test_wrong_format(self):
        self.assertRaises(ValueError, utils.parse_complex_str, '1+1i')


class TestReadGLDCsv(unittest.TestCase):
    """Test utils.read_gld_csv.

    TODO: Once the GridLAB-D tests in test_zip_model.py are wrapped,
    put the .csv files in git, read one or more here, and test.
    """

    def test_1(self):
        df = utils.read_gld_csv('test_zip_1.csv')
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
