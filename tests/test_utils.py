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

    TODO: Test failures, more interesting cases?
    Maybe not worth it, read_gld_csv is really a utility for
    unit testing, and won't be used in the actual application.
    """

    @classmethod
    def setUpClass(cls):
        """Read the file"""
        cls.df = utils.read_gld_csv('test_zip_1.csv')

    def test_shape_0(self):
        self.assertEqual(self.df.shape[0], 41)

    def test_shape_1(self):
        self.assertEqual(self.df.shape[1], 5)

    def test_headings_0(self):
        self.assertEqual(self.df.columns[0], 'timestamp')

    def test_headings_end(self):
        self.assertEqual(self.df.columns[-1], 'measured_reactive_power')

    def test_values_1(self):
        self.assertAlmostEqual(self.df['measured_reactive_power'].iloc[0],
                               5.71375e-07)

    def test_values_2(self):
        val1 = self.df['measured_voltage_2'].iloc[-1]
        val2, _ = utils.parse_complex_str('+139.988-0.00164745d')
        self.assertAlmostEqual(val1.real, val2.real)
        self.assertAlmostEqual(val1.imag, val2.imag)

    def test_values_3(self):
        self.assertEqual(self.df['timestamp'].iloc[-2],
                         '2018-01-01 00:39:00 UTC')


if __name__ == '__main__':
    unittest.main()
